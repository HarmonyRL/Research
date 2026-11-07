"""Share of a harmonization where 1 chord has held for too long."""

from collections.abc import Sequence

from harmonyrl.schema import ChordEvent, Melody

from .spans import chord_spans

_HELD_BARS = 2


def held_share(melody: Melody, events: Sequence[ChordEvent]) -> float:
	"""Share of the 8th notes of `melody` after 1 chord, in any inversion, has held for 2 bars."""
	limit = _HELD_BARS * melody.meter.bar_eighths
	held = start = 0
	previous = None

	for span in chord_spans(melody, events):
		current = None if span.chord is None else span.pitch_classes
		if current != previous:
			start = span.start
		if current is not None:
			held += max(0, span.end - max(span.start, start + limit))

		previous = current

	return held / melody.length
