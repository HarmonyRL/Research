"""Transposition of a segment for the listener: its 1st key moves onto the white keys."""

from dataclasses import replace

from harmonyrl.schema import Key, KeySpan, Mode, Segment

WHITE_TONICS = {
	Mode.MAJOR: 0,
	Mode.MINOR: 9,
	Mode.HARMONIC_MINOR: 9,
	Mode.DORIAN: 2,
	Mode.PHRYGIAN: 4,
	Mode.LYDIAN: 5,
	Mode.MIXOLYDIAN: 7,
	Mode.LOCRIAN: 11,
	Mode.PHRYGIAN_DOMINANT: 4,
}
"""Tonic of each mode on the white keys."""


def to_white_keys(segment: Segment) -> tuple[Segment, int]:
	"""`Segment` moved by the semitones, -6 to +5, that put its 1st key on the white keys,
	and those semitones."""
	first = segment.melody.keys[0].key
	shift = (WHITE_TONICS[first.mode] - first.tonic + 6) % 12 - 6
	melody = replace(
		segment.melody,
		notes=tuple(replace(n, pitch=n.pitch + shift) for n in segment.melody.notes),
		keys=tuple(
			KeySpan(k.start, Key((k.key.tonic + shift) % 12, k.key.mode))
			for k in segment.melody.keys
		),
	)

	return replace(segment, melody=melody), shift
