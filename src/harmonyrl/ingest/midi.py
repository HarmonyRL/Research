"""Import a melody or a lead sheet from a MIDI file."""

from fractions import Fraction
from math import ceil
from pathlib import Path

import pretty_midi

from harmonyrl.schema import Key, KeySpan, LeadSheet, Meter, Mode, Note, SheetPart

from .common import (
	ImportedMelody,
	SheetImport,
	bar_ceiling,
	beat_tempo_from_quarter,
	build_melody,
	estimate_key,
	meter_from_ratio,
	snap,
)


def melody_from_midi(
	path: Path,
	*,
	track: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
) -> ImportedMelody:
	"""Read a monophonic melody from a MIDI file."""
	midi = pretty_midi.PrettyMIDI(str(path))
	instrument = _select_track(midi, track)
	resolved_meter = meter or _meter_from_file(midi)
	tempo = _tempo(midi, resolved_meter)
	notes = _track_notes(midi, instrument)

	end = max((note.offset for note in notes), default=Fraction(0))
	length = max(1, bar_ceiling(end, resolved_meter.bar_eighths))
	declared = _declared_key(midi)
	signature_tonic = None if declared is None else declared.tonic
	resolved_key = (
		key if key is not None else estimate_key(tuple(notes), signature_tonic=signature_tonic)
	)
	melody = build_melody(notes, resolved_meter, (KeySpan(0, resolved_key),), length, tempo=tempo)

	return ImportedMelody(melody=melody, declared_key=declared)


def lead_sheet_from_midi(
	path: Path, *, track: int | None = None, key: Key | None = None
) -> SheetImport:
	"""Read a lead sheet from a MIDI file."""
	midi = pretty_midi.PrettyMIDI(str(path))
	notes = _track_notes(midi, _select_track(midi, track))
	declared = _declared_key(midi)
	resolved_key = key or estimate_key(
		tuple(notes), signature_tonic=None if declared is None else declared.tonic
	)

	starts: list[tuple[int, Meter]] = []
	for signature in sorted(midi.time_signature_changes, key=lambda change: change.time):
		meter = meter_from_ratio(signature.numerator, signature.denominator)
		start = _eighths(midi, signature.time)
		if start.denominator != 1:
			raise ValueError(f"a time signature change at {signature.time}s is off the 8th notes")
		if starts and starts[-1][0] == start:
			starts.pop()
		if not starts or starts[-1][1] != meter:
			starts.append((int(start), meter))
	if not starts or starts[0][0] != 0:
		raise ValueError("no time signature at the start of the MIDI file")

	tempo_times, tempo_values = midi.get_tempo_changes()
	tempos = [
		(_eighths(midi, float(time)), float(bpm))
		for time, bpm in zip(tempo_times, tempo_values, strict=True)
	]

	parts: list[SheetPart] = []
	last = max((note.offset for note in notes), default=Fraction(0))
	for index, (start, meter) in enumerate(starts):
		bar = meter.bar_eighths
		end = (
			starts[index + 1][0]
			if index + 1 < len(starts)
			else start + max(bar, ceil((last - start) / bar) * bar)
		)
		in_effect = [bpm for at, bpm in tempos if at <= start]
		melody = build_melody(
			[
				Note(note.onset - start, min(note.offset, end) - note.onset, note.pitch)
				for note in notes
				if start <= note.onset < end
			],
			meter,
			(KeySpan(0, resolved_key),),
			end - start,
			tempo=beat_tempo_from_quarter(in_effect[-1], meter) if in_effect else None,
		)
		parts.append(SheetPart(melody))

	return SheetImport(LeadSheet(tuple(parts)), declared_key=declared)


def _eighths(midi: pretty_midi.PrettyMIDI, seconds: float) -> Fraction:
	return snap(Fraction(2 * midi.time_to_tick(seconds), midi.resolution))


def _track_notes(midi: pretty_midi.PrettyMIDI, instrument: pretty_midi.Instrument) -> list[Note]:
	notes: list[Note] = []
	for raw_note in instrument.notes:
		onset = _eighths(midi, raw_note.start)
		duration = _eighths(midi, raw_note.end) - onset
		if duration > 0:
			notes.append(Note(onset=onset, duration=duration, pitch=raw_note.pitch))

	return notes


def _declared_key(midi: pretty_midi.PrettyMIDI) -> Key | None:
	"""Major tonic named by the file's first key signature, if any."""
	if not midi.key_signature_changes:
		return None

	_mode, sharps = pretty_midi.key_number_to_mode_accidentals(
		midi.key_signature_changes[0].key_number
	)
	return Key(_MAJOR_TONIC_BY_SHARPS[sharps], Mode.MAJOR)


_MAJOR_TONIC_BY_SHARPS = {
	-7: 11,
	-6: 6,
	-5: 1,
	-4: 8,
	-3: 3,
	-2: 10,
	-1: 5,
	0: 0,
	1: 7,
	2: 2,
	3: 9,
	4: 4,
	5: 11,
	6: 6,
	7: 1,
}
"""Major tonic pitch class for each key-signature sharps count."""


def _select_track(midi: pretty_midi.PrettyMIDI, track: int | None) -> pretty_midi.Instrument:
	if track is not None:
		if not 0 <= track < len(midi.instruments):
			raise ValueError(f"track {track} out of range (0-{len(midi.instruments) - 1})")

		chosen = midi.instruments[track]
		if not chosen.notes:
			raise ValueError(f"track {track} contains no notes")

		return chosen

	instruments = [i for i in midi.instruments if i.notes and not i.is_drum]
	if not instruments:
		raise ValueError("no pitched notes in any MIDI track")
	if len(instruments) == 1:
		return instruments[0]

	return max(instruments, key=_melody_score)


def _melody_score(instrument: pretty_midi.Instrument) -> float:
	"""How melody-like a track is."""
	onsets: dict[float, list[int]] = {}
	for note in instrument.notes:
		onsets.setdefault(round(note.start, 4), []).append(note.pitch)

	pitches = [p for group in onsets.values() for p in group]
	if not pitches:
		return -1.0

	polyphonic = sum(len(group) > 1 for group in onsets.values()) / len(onsets)
	ceiling = max(pitches)
	top_line = max(max(group) for group in onsets.values())
	height = (top_line - 36) / max(1, ceiling - 36) if ceiling > 36 else 0.5

	return len(onsets) * (1.0 - polyphonic) * (0.25 + height)


def _meter_from_file(midi: pretty_midi.PrettyMIDI) -> Meter:
	signatures = midi.time_signature_changes
	if not signatures:
		raise ValueError("no time signature in MIDI; pass meter=")

	distinct = list(dict.fromkeys((s.numerator, s.denominator) for s in signatures))
	if len(distinct) > 1:
		raise ValueError("time signature changes in MIDI; split the song and pass meter=")

	numerator, denominator = distinct[0]
	return meter_from_ratio(numerator, denominator)


def _tempo(midi: pretty_midi.PrettyMIDI, meter: Meter) -> int | None:
	tempos = midi.get_tempo_changes()[1]
	if not len(tempos):
		return None
	return beat_tempo_from_quarter(float(tempos[0]), meter)
