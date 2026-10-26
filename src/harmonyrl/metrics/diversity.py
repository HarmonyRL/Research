"""Diversity and copying: chord n-grams, disagreement between samples, and copies from training."""

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from harmonyrl.schema import ChordEvent, Melody

from .spans import chord_spans


def chord_ngrams(melody: Melody, events: Sequence[ChordEvent], n: int) -> Counter[tuple]:
	"""Counts of runs of `n` consecutive projected chords, no chord included, over events."""
	chords = [span.chord for span in chord_spans(melody, events)]
	return Counter(tuple(chords[i : i + n]) for i in range(len(chords) - n + 1))


def sample_disagreement(melody: Melody, samples: Sequence[Sequence[ChordEvent]]) -> float | None:
	"""Mean share of 8th notes where 2 harmonizations of `melody` differ, over all pairs of samples.

	None with fewer than 2 samples.
	"""
	frames = [
		[span.chord for span in chord_spans(melody, events) for _ in range(span.end - span.start)]
		for events in samples
	]
	shares = [
		sum(x != y for x, y in zip(a, b, strict=True)) / melody.length
		for a, b in combinations(frames, 2)
	]
	return sum(shares) / len(shares) if shares else None


def copy_lengths(
	train: Sequence[tuple[Melody, Sequence[ChordEvent]]],
	harmonizations: Sequence[tuple[Melody, Sequence[ChordEvent]]],
) -> list[int]:
	"""Per harmonization, its longest run of events that also occurs in 1 training harmonization.

	Events match on projected chord and duration. Lengths grow 1 event at a time, so only the
	training runs of the current length are held.
	"""
	vocabulary: dict[tuple, int] = {}

	def tokens(melody: Melody, events: Sequence[ChordEvent]) -> list[int]:
		return [
			vocabulary.setdefault((span.chord, span.end - span.start), len(vocabulary))
			for span in chord_spans(melody, events)
		]

	train_tokens = [tokens(melody, events) for melody, events in train]
	sample_tokens = [tokens(melody, events) for melody, events in harmonizations]
	lengths = [0] * len(sample_tokens)
	alive = list(range(len(sample_tokens)))
	n = 1
	while alive:
		seen = {tuple(t[i : i + n]) for t in train_tokens for i in range(len(t) - n + 1)}
		still = []
		for index in alive:
			t = sample_tokens[index]
			if any(tuple(t[i : i + n]) in seen for i in range(len(t) - n + 1)):
				lengths[index] = n
				still.append(index)
		alive = still
		n += 1
	return lengths
