"""Phrase ends of a melody, read from the melody alone so that all its harmonizations share them."""

from harmonyrl.schema import Melody

PHRASE_BARS = 4


def phrase_ends(melody: Melody) -> tuple[int, ...]:
	"""Positions in 8th notes where a phrase ends, ascending, the last being `melody.length`."""
	size = PHRASE_BARS * melody.meter.bar_eighths
	ends = set(range(_grid_start(melody) + size, melody.length, size))
	ends.update(span.start for span in melody.keys[1:])
	ends.add(melody.length)

	return tuple(sorted(ends))


def phrase_windows(melody: Melody) -> list[tuple[int, int]]:
	"""Start and end in 8th notes of each whole 4-bar window of the grid."""
	size = PHRASE_BARS * melody.meter.bar_eighths
	return [
		(start, start + size)
		for start in range(_grid_start(melody), melody.length - size + 1, size)
	]


def _grid_start(melody: Melody) -> int:
	bar = melody.meter.bar_eighths
	pickup = (
		bool(melody.notes)
		and bar <= 2 * melody.notes[0].onset < 2 * bar
		and (melody.length // bar) % PHRASE_BARS != 0
	)

	return bar if pickup else 0
