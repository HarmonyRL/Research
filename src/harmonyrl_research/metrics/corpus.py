"""All metrics of 1 system over a corpus, against the human accompaniments of the same melodies."""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from harmonyrl_research.schema import ChordEvent, Segment, merge_repeats

from .color import out_of_key_share
from .density import densities
from .distributions import js_distance
from .diversity import chord_ngrams, copy_lengths, sample_disagreement
from .hold import held_share
from .idiom import idiom_counts, idiom_rates
from .structure import key_consistency, phrase_end_patterns, repeated_phrase_agreement
from .yeh import yeh_metrics

NGRAM_SIZES = (2, 3)


def evaluate_corpus(
	segments: Sequence[Segment],
	harmonizations: Sequence[Sequence[Sequence[ChordEvent]]],
	train: Sequence[Segment],
) -> dict[str, float | None]:
	"""Metrics of `harmonizations`, 1 or more per segment, against the human events of `segments`.

	`train` holds the training segments that copies are measured against.
	"""
	if len(segments) != len(harmonizations):
		raise ValueError(f"{len(segments)} segments but {len(harmonizations)} harmonizations")

	if any(not samples for samples in harmonizations):
		raise ValueError("every segment needs a harmonization")

	yeh: dict[str, list[float]] = {}
	idiom = Counter[str]()
	ends = {"system": Counter[str](), "human": Counter[str]()}
	inside = {"system": Counter[str](), "human": Counter[str]()}
	ngrams = {n: {"system": Counter[tuple](), "human": Counter[tuple]()} for n in NGRAM_SIZES}
	repeated = np.zeros(2, dtype=np.int64)
	in_key = np.zeros(2, dtype=np.int64)

	melody_density: list[float] = []
	chord_density: list[float] = []
	disagreement: list[float] = []
	outside: list[float] = []
	held: list[float] = []
	systems: list[tuple[Segment, tuple[ChordEvent, ...]]] = []

	for segment, samples in zip(segments, harmonizations, strict=True):
		melody = segment.melody
		human = merge_repeats(segment.events, melody)
		merged = [merge_repeats(events, melody) for events in samples]

		for events in merged:
			for name, value in yeh_metrics(melody, events).items():
				values = yeh.setdefault(name, [])
				if value is not None:
					values.append(value)

			idiom += idiom_counts(melody, events)
			repeated += repeated_phrase_agreement(melody, events)
			in_key += key_consistency(melody, events)
			outside.append(out_of_key_share(melody, events))
			held.append(held_share(melody, events))

			notes, chords = densities(melody, events)
			melody_density.append(notes)
			chord_density.append(chords)
			systems.append((segment, events))

		for side, side_samples in (("system", merged), ("human", [human])):
			for side_events in side_samples:
				side_ends, side_inside = phrase_end_patterns(melody, side_events)
				ends[side].update(side_ends)
				inside[side].update(side_inside)

				for n in NGRAM_SIZES:
					ngrams[n][side].update(chord_ngrams(melody, side_events, n))

		spread = sample_disagreement(melody, merged)
		if spread is not None:
			disagreement.append(spread)

	copies = copy_lengths(
		[(s.melody, merge_repeats(s.events, s.melody)) for s in train],
		[(s.melody, events) for s, events in systems],
	)

	return {
		**{name: float(np.mean(values)) if values else None for name, values in yeh.items()},
		**idiom_rates(idiom),
		"repeated_phrase_agreement": float(repeated[0] / repeated[1]) if repeated[1] else None,
		"key_consistency": float(in_key[0] / in_key[1]) if in_key[1] else None,
		"phrase_end_js": js_distance(ends["system"], ends["human"]),
		"phrase_inside_js": _js_or_none(inside["system"], inside["human"]),
		**{f"chord_{n}gram_js": _js_or_none(c["system"], c["human"]) for n, c in ngrams.items()},
		"copy_length_mean": float(np.mean(copies)),
		"copy_length_max": float(max(copies)),
		"chords_per_bar": float(np.mean(chord_density)),
		"density_correlation": float(np.corrcoef(melody_density, chord_density)[0, 1]),
		"out_of_key_share": float(np.mean(outside)),
		"held_share": float(np.mean(held)),
		"sample_disagreement": float(np.mean(disagreement)) if disagreement else None,
	}


def _js_or_none(system: Counter, human: Counter) -> float | None:
	"""Distance between 2 distributions, or None when either is empty, as for 1-event segments."""
	return js_distance(system, human) if system and human else None
