"""Fixed voicing of chord events for audio rendering.

Every system is rendered through these rules, so the voicing never varies between
experiments. The bass sits in the octave starting at C2 and the remaining chord tones
are packed in ascending order from C3 upward.
"""

from dataclasses import dataclass
from fractions import Fraction

from harmonyrl.schema import Segment, chord_pitch_classes

BASS_BOTTOM = 36
"""MIDI note of the bass voice octave (C2)."""

UPPER_BOTTOM = 48
"""Lowest MIDI note allowed for upper voices (C3)."""

MELODY_PROGRAM = 0
"""General MIDI program of the melody voice: Acoustic Grand."""

CHORD_PROGRAM = 24
"""General MIDI program of chord voices: Acoustic Guitar (nylon)."""

MELODY_VELOCITY = 90
CHORD_VELOCITY = 70


@dataclass(frozen=True)
class VoicedNote:
	"""One rendered note. `onset` and `duration` are in 8th notes."""

	onset: Fraction
	duration: Fraction
	pitch: int
	program: int
	velocity: int


def melody_notes(segment: Segment) -> tuple[VoicedNote, ...]:
	return tuple(
		VoicedNote(note.onset, note.duration, note.pitch, MELODY_PROGRAM, MELODY_VELOCITY)
		for note in segment.melody.notes
	)


def chord_notes(segment: Segment) -> tuple[VoicedNote, ...]:
	"""Deterministic voicing of every chord event in the segment."""
	position = Fraction(0)
	notes: list[VoicedNote] = []
	for event in segment.events:
		if event.chord is not None:
			key = segment.melody.key_at(position)
			bass_pc, pitch_classes = chord_pitch_classes(event.chord, key)
			for pitch in _voice_chord(bass_pc, pitch_classes):
				notes.append(
					VoicedNote(
						position, Fraction(event.duration), pitch, CHORD_PROGRAM, CHORD_VELOCITY
					)
				)
		position += event.duration
	return tuple(notes)


def _voice_chord(bass_pc: int, pitch_classes: frozenset[int]) -> tuple[int, ...]:
	pitches = [BASS_BOTTOM + bass_pc]
	for pc in sorted(pitch_classes - {bass_pc}):
		pitch = UPPER_BOTTOM + (pc - UPPER_BOTTOM) % 12
		while pitch <= pitches[-1]:
			pitch += 12
		pitches.append(pitch)
	return tuple(pitches)
