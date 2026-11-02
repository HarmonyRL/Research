"""MIDI rendering of melodies with chord events, using the fixed voicing."""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pretty_midi

from harmonyrl.schema import ChordEvent, Key, Melody

from .symbol import midi_key_number
from .voicing import check_tiling, chord_notes, melody_notes

DEFAULT_TEMPO = 120
"""Tempo in BPM of a melody whose tempo is unknown."""

TICKS_PER_QUARTERBEAT = 480
"""MIDI resolution. Divisible by 48, so every note length in the dataset is exact."""


def write_midi(
	melody: Melody,
	events: Sequence[ChordEvent],
	path: Path,
) -> None:
	"""Write a melody with chord events as a MIDI file, one instrument per voice group.

	The melody plays at its tempo, or at `DEFAULT_TEMPO` when that is unknown.
	"""
	check_tiling(melody, events)

	tempo = DEFAULT_TEMPO if melody.tempo is None else melody.tempo
	quarter_bpm = Fraction(tempo * (3 if melody.meter.beat_unit == 8 else 2), 2)
	eighth = Fraction(30, quarter_bpm)

	midi = pretty_midi.PrettyMIDI(
		resolution=TICKS_PER_QUARTERBEAT, initial_tempo=float(quarter_bpm)
	)
	midi.time_signature_changes.append(
		pretty_midi.TimeSignature(melody.meter.beats_per_bar, melody.meter.beat_unit, 0.0)
	)
	previous_key: Key | None = None

	for span in melody.keys:
		if span.key != previous_key:
			midi.key_signature_changes.append(
				pretty_midi.KeySignature(midi_key_number(span.key), float(span.start * eighth))
			)
			previous_key = span.key

	for notes in (melody_notes(melody), chord_notes(melody, events)):
		if not notes:
			continue

		instrument = pretty_midi.Instrument(program=notes[0].program)

		for note in notes:
			instrument.notes.append(
				pretty_midi.Note(
					velocity=note.velocity,
					pitch=note.pitch,
					start=float(note.onset * eighth),
					end=float((note.onset + note.duration) * eighth),
				)
			)

		midi.instruments.append(instrument)

	midi.write(str(path))
