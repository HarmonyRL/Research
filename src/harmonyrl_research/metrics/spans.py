"""Chord events laid out in time, by default with chords projected onto the model vocabulary."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from harmonyrl_research.schema import (
	ChordEvent,
	Key,
	Melody,
	RawChord,
	chord_pitch_classes,
	project,
)


@dataclass(frozen=True)
class ChordSpan:
	"""A chord event from `start` to `end` in 8th notes."""

	start: int
	end: int
	chord: RawChord | None
	key: Key
	bass: int | None
	pitch_classes: frozenset[int]


def chord_spans(
	melody: Melody, events: Sequence[ChordEvent], *, projected: bool = True
) -> list[ChordSpan]:
	"""Spans of `events`, which must tile `melody`.

	With `projected` off, chords stay as given, for statistics of the source data itself.
	"""
	total = sum(event.duration for event in events)
	if total != melody.length:
		raise ValueError(f"event durations sum to {total}, melody length is {melody.length}")

	spans: list[ChordSpan] = []
	start = 0
	for event in events:
		key = melody.key_at(start)

		if event.chord is None:
			spans.append(ChordSpan(start, start + event.duration, None, key, None, frozenset()))
		else:
			chord = project(event.chord, key) if projected else event.chord
			bass, pitch_classes = chord_pitch_classes(chord, key)
			spans.append(ChordSpan(start, start + event.duration, chord, key, bass, pitch_classes))

		start += event.duration

	return spans


def span_at(spans: Sequence[ChordSpan], position: Fraction | int) -> ChordSpan:
	"""The span sounding at `position`, which must lie within the spans."""
	return spans[bisect_right([span.start for span in spans], position) - 1]
