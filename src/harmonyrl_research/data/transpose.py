"""Transposition of a segment by any number of semitones."""

from dataclasses import replace

from harmonyrl_research.schema import Key, KeySpan, Segment


def transpose(segment: Segment, semitones: int) -> Segment:
	"""`segment` moved by `semitones`. Chords are relative to the local key,
	so only notes and keys move."""
	melody = segment.melody
	return replace(
		segment,
		melody=replace(
			melody,
			notes=tuple(replace(n, pitch=n.pitch + semitones) for n in melody.notes),
			keys=tuple(
				KeySpan(k.start, Key((k.key.tonic + semitones) % 12, k.key.mode))
				for k in melody.keys
			),
		),
	)
