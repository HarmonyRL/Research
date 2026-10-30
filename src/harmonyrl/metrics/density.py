"""How densely the melody and the chords of a segment fill its bars."""

from collections.abc import Sequence

from harmonyrl.schema import ChordEvent, Melody


def densities(melody: Melody, events: Sequence[ChordEvent]) -> tuple[float, float]:
	"""Melody notes and chord events per bar. No chord is not counted as a chord event."""
	bars = melody.length / melody.meter.bar_eighths
	chords = sum(event.chord is not None for event in events)

	return len(melody.notes) / bars, chords / bars
