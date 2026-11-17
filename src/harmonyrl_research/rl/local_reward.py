"""Per-note rewards, the local reward baseline against the learned reward."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise

import torch

from harmonyrl_research.metrics import ChordSpan, chord_spans, span_at
from harmonyrl_research.schema import Melody, Meter, Segment


@dataclass(frozen=True)
class _Step:
	start: Fraction
	duration: Fraction
	pitch: int | None
	chord: ChordSpan


def local_rewards(samples: Sequence[Segment], references: Sequence[Segment]) -> torch.Tensor:
	"""Mean step reward of each sample, against the human harmonization of the same melody."""
	return torch.tensor(
		[_reward(s, r) for s, r in zip(samples, references, strict=True)], dtype=torch.float32
	)


def _reward(sample: Segment, reference: Segment) -> float:
	melody = sample.melody
	human = chord_spans(melody, reference.events)
	steps = _steps(melody, chord_spans(melody, sample.events), human)
	total = 0.0

	for i, step in enumerate(steps):
		previous = steps[i - 1].chord if i else None

		total += _ground_truth(step.chord, span_at(human, step.start))
		total += _harmony(steps, i)
		total += _progression(step, previous, melody.meter)

	return total / len(steps)


def _steps(melody: Melody, spans: Sequence[ChordSpan], human: Sequence[ChordSpan]) -> list[_Step]:
	"""Notes and rests of `melody` in time order with their chords in `spans`, cut where `human`
	starts a chord and where `spans` change chord.
	"""
	boundaries = {Fraction(s.start) for s in human}
	boundaries |= {Fraction(b.start) for a, b in pairwise(spans) if a.chord != b.chord}
	pieces: list[tuple[Fraction, Fraction, int | None]] = []
	position = Fraction(0)

	for note in melody.notes:
		if note.onset > position:
			pieces.append((position, note.onset, None))

		pieces.append((note.onset, note.offset, note.pitch))
		position = note.offset

	if position < melody.length:
		pieces.append((position, Fraction(melody.length), None))

	steps: list[_Step] = []
	for start, end, pitch in pieces:
		cuts = sorted({start, end} | {b for b in boundaries if start < b < end})
		steps.extend(_Step(a, b - a, pitch, span_at(spans, a)) for a, b in pairwise(cuts))

	return steps


def _ground_truth(chord: ChordSpan, truth: ChordSpan) -> float:
	"""Pitch classes shared with the human chord, less the mismatch of no chord, bass,
	and pitch classes.
	"""
	mismatch = (
		((chord.chord is None) != (truth.chord is None))
		+ (chord.bass != truth.bass)
		+ len(chord.pitch_classes ^ truth.pitch_classes)
	)

	return len(chord.pitch_classes & truth.pitch_classes) - mismatch


def _harmony(steps: Sequence[_Step], i: int) -> float:
	"""Consonance of the chord of step `i` with the notes of the steps next to it and itself."""
	tones = sorted(steps[i].chord.pitch_classes)
	if not tones:
		return 0.0

	notes = [(s.pitch, s.duration) for s in steps[max(i - 1, 0) : i + 2] if s.pitch is not None]
	weight = sum(duration for _, duration in notes)
	between = 0.0

	if weight:
		between = float(
			sum(_consonance(t, pitch % 12) * duration for pitch, duration in notes for t in tones)
			/ (weight * len(tones))
		)

	within = sum(_consonance(a, b) for a, b in combinations(tones, 2)) / len(tones) ** 2
	return between + within


def _progression(step: _Step, previous: ChordSpan | None, meter: Meter) -> int:
	"""-1 for a chord held into a bar's 1st step, +1 for one held inside a bar,
	and -1 when no tone is shared with the chord before.
	"""
	if previous is None:
		return 0

	repetition = 0
	if step.chord.chord == previous.chord:
		repetition = -1 if step.start % meter.bar_eighths == 0 else 1

	tones, before = step.chord.pitch_classes, previous.pitch_classes
	disjoint = bool(tones) and bool(before) and not tones & before

	return repetition - disjoint


def _consonance(a: int, b: int) -> int:
	"""Consonance of 2 pitch classes, from the difference of their numbers."""
	return {0: 10, 5: 8, 7: 8, 4: 6, 8: 6, 3: 5, 9: 5, 2: 3, 10: 3, 1: 1, 11: 1}.get(abs(a - b), 0)
