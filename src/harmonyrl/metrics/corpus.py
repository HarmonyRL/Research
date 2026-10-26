"""All metrics of 1 system over a corpus, against the human accompaniments of the same melodies.

Per-segment metrics are averaged over the segments where they are defined; counts are pooled
before taking rates; distributions are compared with the human ones by Jensen-Shannon distance.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from harmonyrl.schema import ChordEvent, Segment

from .distributions import js_distance
from .diversity import chord_ngrams, copy_lengths
from .idiom import idiom_counts, idiom_rates
from .structure import key_consistency, phrase_end_patterns, repeated_phrase_agreement
from .yeh import yeh_metrics

NGRAM_SIZES = (2, 3)


def evaluate_corpus(
	segments: Sequence[Segment],
	harmonizations: Sequence[Sequence[ChordEvent]],
	train: Sequence[Segment],
) -> dict[str, float | None]:
	"""Metrics of `harmonizations`, 1 per segment, against the human events of `segments`.

	`train` holds the training segments that copies are measured against.
	"""
	if len(segments) != len(harmonizations):
		raise ValueError(f"{len(segments)} segments but {len(harmonizations)} harmonizations")

	yeh: dict[str, list[float]] = {}
	idiom = Counter[str]()
	ends = {"system": Counter[str](), "human": Counter[str]()}
	inside = {"system": Counter[str](), "human": Counter[str]()}
	ngrams = {n: {"system": Counter[tuple](), "human": Counter[tuple]()} for n in NGRAM_SIZES}
	repeated = np.zeros(2, dtype=np.int64)
	in_key = np.zeros(2, dtype=np.int64)

	for segment, events in zip(segments, harmonizations, strict=True):
		melody = segment.melody
		for name, value in yeh_metrics(melody, events).items():
			values = yeh.setdefault(name, [])
			if value is not None:
				values.append(value)
		idiom += idiom_counts(melody, events)
		repeated += repeated_phrase_agreement(melody, events)
		in_key += key_consistency(melody, events)
		for side, side_events in (("system", events), ("human", segment.events)):
			side_ends, side_inside = phrase_end_patterns(melody, side_events)
			ends[side].update(side_ends)
			inside[side].update(side_inside)
			for n in NGRAM_SIZES:
				ngrams[n][side].update(chord_ngrams(melody, side_events, n))

	copies = copy_lengths(
		[(s.melody, s.events) for s in train],
		[(s.melody, events) for s, events in zip(segments, harmonizations, strict=True)],
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
	}


def _js_or_none(system: Counter, human: Counter) -> float | None:
	"""Distance between 2 distributions, or None when either is empty, as for 1-event segments."""
	return js_distance(system, human) if system and human else None
