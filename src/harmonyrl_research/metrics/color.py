"""Share of a harmonization played by chords outside the key."""

from collections.abc import Sequence

from harmonyrl_research.schema import ChordEvent, Key, Melody, Mode, merge_repeats

from .spans import chord_spans

_MINOR_DEGREES = frozenset({0, 2, 3, 5, 7, 8, 9, 10, 11})
"""Natural, harmonic, and melodic minor together, as all 3 are ordinary in a minor key."""


def _in_key(key: Key) -> set[int]:
	minor = key.mode in (Mode.MINOR, Mode.HARMONIC_MINOR)
	degrees = _MINOR_DEGREES if minor else key.mode.scale

	return {(key.tonic + degree) % 12 for degree in degrees}


def out_of_key_share(melody: Melody, events: Sequence[ChordEvent]) -> float:
	"""Share of the 8th notes of `melody` under a chord with a pitch class outside the local key."""
	outside = sum(
		span.end - span.start
		for span in chord_spans(melody, merge_repeats(events, melody))
		if span.chord is not None and span.pitch_classes - _in_key(span.key)
	)

	return outside / melody.length
