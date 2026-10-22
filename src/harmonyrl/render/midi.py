"""MIDI rendering of segments with the fixed voicing."""

from fractions import Fraction
from pathlib import Path

import pretty_midi

from harmonyrl.schema import Segment

from .voicing import chord_notes, melody_notes

DEFAULT_TEMPO = 120
"""Quarter-note tempo used when a segment has no recorded source tempo."""

TICKS_PER_QUARTERBEAT = 480
"""MIDI resolution. Divisible by 48, so every note length in the dataset is exact."""


def write_midi(segment: Segment, path: Path, tempo: int = DEFAULT_TEMPO) -> None:
	"""Write `segment` as a MIDI file with one instrument per voice group."""
	eighth = Fraction(30, tempo)
	midi = pretty_midi.PrettyMIDI(resolution=TICKS_PER_QUARTERBEAT, initial_tempo=float(tempo))
	for notes in (melody_notes(segment), chord_notes(segment)):
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
