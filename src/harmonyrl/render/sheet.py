"""Writing a lead sheet, its parts joined, to any supported file."""

import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

import music21
import pretty_midi

from harmonyrl.schema import Key, LeadSheet, Meter

from .audio import midi_to_wav, wav_to_mp3
from .convert import musicxml_to_formats
from .midi import DEFAULT_TEMPO, TICKS_PER_QUARTERBEAT
from .score import melody_to_score, write_score
from .symbol import midi_key_number
from .voicing import chord_notes, melody_notes


def write_lead_sheet(
	sheet: LeadSheet, path: Path, *, mscore: Path | None = None, soundfont: Path | None = None
) -> None:
	"""Write `sheet` as a score, MIDI, or recording, chosen by file suffix."""
	suffix = path.suffix.lower()

	if suffix in (".musicxml", ".xml"):
		write_score(_score(sheet), path)
	elif suffix in (".mscz", ".mscx", ".pdf"):
		with tempfile.TemporaryDirectory() as tmp:
			musicxml = Path(tmp) / "sheet.musicxml"
			write_score(_score(sheet), musicxml)
			musicxml_to_formats(musicxml, [path], mscore or _mscore())
	elif suffix in (".mid", ".midi"):
		_write_midi(sheet, path)
	elif suffix in (".wav", ".mp3"):
		if soundfont is None:
			raise ValueError("writing audio needs soundfont=")
		with tempfile.TemporaryDirectory() as tmp:
			midi = Path(tmp) / "sheet.mid"
			_write_midi(sheet, midi)

			wav = path if suffix == ".wav" else Path(tmp) / "sheet.wav"
			midi_to_wav(midi, wav, soundfont)

			if suffix == ".mp3":
				wav_to_mp3(wav, path)
	else:
		raise ValueError(f"unsupported file type {suffix!r}")


def _score(sheet: LeadSheet) -> music21.stream.Score:
	"""The parts of `sheet` on 1 staff, each starting after a double barline."""
	part = music21.stream.Part()
	part.partName = "Melody"
	measures: list[music21.stream.Measure] = []

	previous_meter: Meter | None = None
	previous_key: Key | None = None
	previous_tempo: int | None = None

	for sheet_part in sheet.parts:
		melody = sheet_part.melody
		source = melody_to_score(melody, sheet_part.chords).parts[0]
		added = list(source.getElementsByClass(music21.stream.Measure))
		first = added[0]

		if measures:
			measures[-1].rightBarline = music21.bar.Barline("double")
			_drop(first, music21.clef.Clef)
			if melody.meter == previous_meter:
				_drop(first, music21.meter.TimeSignature)
			if melody.keys[0].key == previous_key:
				_drop(first, music21.key.KeySignature)
			if melody.tempo == previous_tempo:
				_drop(first, music21.tempo.MetronomeMark)

		if melody.section is not None:
			label = melody.section.value.replace("_", "-").title()
			first.insert(0, music21.expressions.RehearsalMark(label))

		for measure in added:
			source.remove(measure)
			measure.number = len(measures) + 1
			part.append(measure)
			measures.append(measure)

		previous_meter, previous_key = melody.meter, melody.keys[-1].key
		previous_tempo = melody.tempo

	score = music21.stream.Score()
	if sheet.title is not None:
		score.metadata = music21.metadata.Metadata(title=sheet.title)
	score.insert(0, part)
	return score


def _drop(measure: music21.stream.Measure, kind: type[music21.base.Music21Object]) -> None:
	for element in list(measure.getElementsByClass(kind)):
		measure.remove(element)


def _write_midi(sheet: LeadSheet, path: Path) -> None:
	"""MIDI of `sheet` at the tempo of its 1st part, or `DEFAULT_TEMPO` when that is unknown."""
	first = sheet.parts[0].melody
	tempo = DEFAULT_TEMPO if first.tempo is None else first.tempo
	quarter_bpm = Fraction(tempo * (3 if first.meter.beat_unit == 8 else 2), 2)
	eighth = Fraction(30, quarter_bpm)

	midi = pretty_midi.PrettyMIDI(
		resolution=TICKS_PER_QUARTERBEAT, initial_tempo=float(quarter_bpm)
	)
	instruments: dict[int, pretty_midi.Instrument] = {}
	start = 0
	previous_meter: Meter | None = None
	previous_key: Key | None = None

	for sheet_part in sheet.parts:
		melody = sheet_part.melody
		if melody.meter != previous_meter:
			midi.time_signature_changes.append(
				pretty_midi.TimeSignature(
					melody.meter.beats_per_bar, melody.meter.beat_unit, float(start * eighth)
				)
			)
			previous_meter = melody.meter

		for span in melody.keys:
			if span.key != previous_key:
				midi.key_signature_changes.append(
					pretty_midi.KeySignature(
						midi_key_number(span.key), float((start + span.start) * eighth)
					)
				)
				previous_key = span.key

		notes = melody_notes(melody)
		if sheet_part.chords is not None:
			notes += chord_notes(melody, sheet_part.chords)

		for note in notes:
			instrument = instruments.setdefault(
				note.program, pretty_midi.Instrument(program=note.program)
			)
			instrument.notes.append(
				pretty_midi.Note(
					velocity=note.velocity,
					pitch=note.pitch,
					start=float((start + note.onset) * eighth),
					end=float((start + note.onset + note.duration) * eighth),
				)
			)

		start += melody.length

	midi.instruments.extend(instruments.values())
	midi.write(str(path))


def _mscore() -> Path:
	found = shutil.which("mscore")
	if found is None:
		raise ValueError("MuseScore (mscore) is not on PATH; pass mscore=")
	return Path(found)
