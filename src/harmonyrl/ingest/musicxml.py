"""Import a melody from a MusicXML file."""

from fractions import Fraction
from pathlib import Path

import music21

from harmonyrl.schema import Key, KeySpan, Meter, Mode, Note

from .common import (
	ImportedMelody,
	bar_ceiling,
	beat_tempo_from_quarter,
	build_melody,
	estimate_key,
	meter_from_ratio,
	snap,
)


def melody_from_musicxml(
	path: Path,
	*,
	part: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
) -> ImportedMelody:
	"""Read a monophonic melody from a MusicXML file."""
	score = music21.converter.parse(str(path))
	voice = _select_part(score, part)
	resolved_meter = meter or _meter_from_score(score, voice)
	tempo = _tempo_from_score(score, resolved_meter)
	pickup = _pickup_eighths(voice)

	notes: list[Note] = []
	for element in voice.recurse().notes:
		# music21 exposes chord symbols through the notes iterator; they are
		# annotation, not melody.
		if isinstance(element, music21.harmony.Harmony):
			continue
		raw_duration = Fraction(element.duration.quarterLength) * 2
		if raw_duration <= 0:
			continue
		pitch = (
			element.pitch.midi
			if isinstance(element, music21.note.Note)
			else max(p.midi for p in element.pitches)
		)
		raw_onset = pickup + Fraction(element.getOffsetInHierarchy(voice)) * 2
		onset = snap(raw_onset)
		end_pos = snap(raw_onset + raw_duration)
		if _is_tied_continuation(element) and _extend_tied_note(notes, pitch, onset, end_pos):
			continue
		duration = end_pos - onset
		if duration > 0:
			notes.append(Note(onset, duration, pitch))

	declared = _declared_key(score, voice)
	end = max((note.offset for note in notes), default=Fraction(0))
	length = max(1, bar_ceiling(end, resolved_meter.bar_eighths))
	signature_tonic = None if declared is None else declared.tonic
	resolved_key = (
		key if key is not None else estimate_key(tuple(notes), signature_tonic=signature_tonic)
	)
	melody = build_melody(notes, resolved_meter, (KeySpan(0, resolved_key),), length, tempo=tempo)

	return ImportedMelody(melody=melody, declared_key=declared)


def _pickup_eighths(voice: music21.stream.Stream) -> Fraction:
	"""Leading rest in 8th notes needed before an incomplete pickup measure so
	position 0 lands on a barline, as the `Melody` contract requires."""
	measures = list(voice.getElementsByClass(music21.stream.Measure))
	if measures and measures[0].paddingLeft > 0:
		return snap(Fraction(measures[0].paddingLeft) * 2)

	return Fraction(0)


def _is_tied_continuation(element: music21.note.NotRest) -> bool:
	tie = element.tie
	return tie is not None and tie.type in ("continue", "stop")


def _extend_tied_note(notes: list[Note], pitch: int, onset: Fraction, end_pos: Fraction) -> bool:
	"""Extend the matching open note in `notes` when a tied continuation arrives."""
	for index in range(len(notes) - 1, -1, -1):
		previous = notes[index]
		if previous.offset < onset:
			break
		if previous.pitch == pitch and previous.offset == onset and end_pos > previous.onset:
			notes[index] = Note(previous.onset, end_pos - previous.onset, previous.pitch)
			return True

	return False


def _select_part(score: music21.stream.Stream, part: int | None) -> music21.stream.Stream:
	parts: list[music21.stream.Stream] = (
		list(score.parts) if isinstance(score, music21.stream.Score) and score.parts else [score]
	)
	if part is not None:
		if not 0 <= part < len(parts):
			raise ValueError(f"part {part} out of range (0-{len(parts) - 1})")

		return parts[part]

	def melody_key(candidate: music21.stream.Stream) -> tuple[float, int]:
		pitches = [
			n.pitch.midi if isinstance(n, music21.note.Note) else max(p.midi for p in n.pitches)
			for n in candidate.recurse().notes
			if not isinstance(n, music21.harmony.Harmony)
		]
		return (sum(pitches) / len(pitches) if pitches else 0.0, len(pitches))

	return max(parts, key=melody_key)


def _meter_from_score(score: music21.stream.Stream, voice: music21.stream.Stream) -> Meter:
	signatures = list(voice.recurse().getElementsByClass(music21.meter.TimeSignature))
	if not signatures:
		signatures = list(score.recurse().getElementsByClass(music21.meter.TimeSignature))
	if not signatures:
		raise ValueError("no time signature in MusicXML; pass meter=")

	distinct = list(dict.fromkeys((s.numerator, s.denominator) for s in signatures))
	if len(distinct) > 1:
		raise ValueError("time signature changes in MusicXML; split the song and pass meter=")

	numerator, denominator = distinct[0]
	return meter_from_ratio(numerator, denominator)


def _tempo_from_score(score: music21.stream.Stream, meter: Meter) -> int | None:
	for mark in score.recurse().getElementsByClass(music21.tempo.MetronomeMark):
		quarter_bpm = mark.getQuarterBPM()
		if quarter_bpm is not None and quarter_bpm > 0:
			return beat_tempo_from_quarter(float(quarter_bpm), meter)

	return None


def _declared_key(score: music21.stream.Stream, voice: music21.stream.Stream) -> Key | None:
	"""Major tonic named by the file's first key signature, if any."""
	signatures = list(voice.recurse().getElementsByClass(music21.key.KeySignature))
	if not signatures:
		signatures = list(score.recurse().getElementsByClass(music21.key.KeySignature))

	for signature in signatures:
		try:
			named = signature.asKey()
		except music21.exceptions21.Music21Exception:
			return None

		return Key(named.tonic.pitchClass, Mode.MAJOR)

	return None
