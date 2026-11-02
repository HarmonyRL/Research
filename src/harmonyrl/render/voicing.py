"""Fixed voicing of chord events for audio rendering."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations, product

from harmonyrl.schema import ChordEvent, Melody, chord_pitch_classes

UPPER_FLOOR = 48
"""Lowest MIDI note allowed for upper voices (C3), keeping them clear of the bass."""

_UPPER_CEILING = 83
"""Highest candidate note for an upper voice (B5); real placements stay far below."""

# Low interval limits.
_LOW_LIMIT = {
	1: 52,
	2: 51,
	3: 48,
	4: 46,
	5: 46,
	6: 46,
	7: 34,
	8: 43,
	9: 42,
	10: 41,
	11: 41,
	13: 40,
	14: 39,
	15: 36,
	16: 34,
}

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


def check_tiling(melody: Melody, events: Sequence[ChordEvent]) -> None:
	"""Verify the events tile the melody, which every rendering input must satisfy."""
	total = sum(event.duration for event in events)
	if total != melody.length:
		raise ValueError(f"event durations sum to {total}, melody length is {melody.length}")


def melody_notes(melody: Melody) -> tuple[VoicedNote, ...]:
	return tuple(
		VoicedNote(note.onset, note.duration, note.pitch, MELODY_PROGRAM, MELODY_VELOCITY)
		for note in melody.notes
	)


def chord_notes(melody: Melody, events: Sequence[ChordEvent]) -> tuple[VoicedNote, ...]:
	"""Deterministic voicing of every chord event over the melody."""
	position = Fraction(0)
	notes: list[VoicedNote] = []

	for event in events:
		if event.chord is not None:
			key = melody.key_at(position)
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
	uppers = tuple(sorted(pitch_classes - {bass_pc}))

	if bass_pc <= 4:
		high = _voice(bass_pc + 36, uppers)
		low = _voice(bass_pc + 24, uppers)

		return low if sum(low[1:]) < sum(high[1:]) else high

	return _voice(bass_pc + 24, uppers)


@cache
def _voice(bass: int, uppers: tuple[int, ...]) -> tuple[int, ...]:
	"""The lowest open voicing of `uppers` above `bass` that respects the limits."""
	candidates = [range(UPPER_FLOOR + pc, _UPPER_CEILING + 1, 12) for pc in uppers]
	best: tuple[int, ...] | None = None

	for combo in product(*candidates):
		notes = (bass, *combo)
		if all(
			hi - lo not in _LOW_LIMIT or lo >= _LOW_LIMIT[hi - lo]
			for lo, hi in combinations(notes, 2)
		) and (best is None or (sum(combo), combo) < (sum(best), best)):
			best = combo

	if best is None:
		raise ValueError(f"no voicing within the low interval limits for bass {bass}, {uppers}")

	return (bass, *best)
