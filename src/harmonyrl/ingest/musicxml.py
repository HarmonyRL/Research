"""Import a melody or a lead sheet from a MusicXML file."""

from dataclasses import replace
from fractions import Fraction
from math import ceil
from pathlib import Path

import music21

from harmonyrl.render import DEFAULT_TITLE
from harmonyrl.schema import (
	MAX_DURATION,
	ChordEvent,
	Key,
	KeySpan,
	LeadSheet,
	Meter,
	Mode,
	Note,
	Section,
	SheetPart,
	section_from_label,
)

from .chords import chord_from_symbol
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

_REPEATS = (music21.bar.Repeat, music21.repeat.RepeatExpression)

_PLACEHOLDER_TITLE = (DEFAULT_TITLE, "Music21 Fragment")
"""Titles that stand for none."""


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
	notes = _voice_notes(voice, _pickup_eighths(voice))

	declared = _declared_key(score, voice)
	end = max((note.offset for note in notes), default=Fraction(0))
	length = max(1, bar_ceiling(end, resolved_meter.bar_eighths))

	signature_tonic = None if declared is None else declared.tonic
	resolved_key = (
		key if key is not None else estimate_key(tuple(notes), signature_tonic=signature_tonic)
	)
	melody = build_melody(notes, resolved_meter, (KeySpan(0, resolved_key),), length, tempo=tempo)

	return ImportedMelody(melody=melody, declared_key=declared)


def lead_sheet_from_musicxml(
	path: Path, *, part: int | None = None, key: Key | None = None
) -> SheetImport:
	"""Read a lead sheet from a MusicXML file."""
	score = music21.converter.parse(str(path))
	voice = _select_part(score, part)
	pickup = _pickup_eighths(voice)
	notes = _voice_notes(voice, pickup)
	declared = _declared_key(score, voice)
	resolved_key = key or estimate_key(
		tuple(notes), signature_tonic=None if declared is None else declared.tonic
	)

	starts = _part_starts(voice, pickup)
	measures = list(voice.getElementsByClass(music21.stream.Measure))
	written_end = (
		pickup + _eighths(measures[-1], voice) + Fraction(measures[-1].duration.quarterLength) * 2
	)
	tempos = _tempos(score, pickup)
	symbols = _chord_symbols(voice, pickup)

	parts: list[SheetPart] = []
	approximated: list[int] = []
	first_bar = 0

	for index, (start, meter, section) in enumerate(starts):
		bar = meter.bar_eighths
		if index + 1 < len(starts):
			end = starts[index + 1][0]
		else:
			last = max([written_end, *(note.offset for note in notes)])
			end = start + ceil((last - start) / bar) * bar

		melody = build_melody(
			[
				Note(note.onset - start, min(note.offset, end) - note.onset, note.pitch)
				for note in notes
				if start <= note.onset < end
			],
			meter,
			(KeySpan(0, resolved_key),),
			end - start,
			tempo=_tempo_at(tempos, start, meter),
		)
		chords, inexact = _part_chords(symbols, start, end, resolved_key)
		parts.append(SheetPart(replace(melody, section=section), chords))
		approximated += [first_bar + (at - start) // bar for at in inexact]
		first_bar += (end - start) // bar

	warnings = ()
	if _has_repeats(score):
		warnings = (
			"repeats, endings, and jumps are ignored: the melody is read once as written, "
			"and the output has no repeat signs",
		)

	title = score.metadata.bestTitle if score.metadata is not None else None
	return SheetImport(
		LeadSheet(tuple(parts), title=None if title in (None, "", *_PLACEHOLDER_TITLE) else title),
		declared_key=declared,
		approximated=tuple(sorted(set(approximated))),
		warnings=warnings,
	)


def _voice_notes(voice: music21.stream.Stream, pickup: Fraction) -> list[Note]:
	"""Notes of `voice` in 8th notes from the barline before its pickup, ties joined."""
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

	return notes


def _eighths(element: music21.base.Music21Object, voice: music21.stream.Stream) -> Fraction:
	return Fraction(element.getOffsetInHierarchy(voice)) * 2


def _part_starts(
	voice: music21.stream.Stream, pickup: Fraction
) -> list[tuple[int, Meter, Section | None]]:
	"""Where each part starts, in 8th notes, with its meter and the section its mark names."""
	starts: list[tuple[int, Meter, Section | None]] = []
	meter: Meter | None = None

	for index, measure in enumerate(voice.getElementsByClass(music21.stream.Measure)):
		signature = measure.timeSignature
		changed = signature is not None and (
			meter is None
			or (signature.numerator, signature.denominator)
			!= (meter.beats_per_bar, meter.beat_unit)
		)
		if changed:
			meter = meter_from_ratio(signature.numerator, signature.denominator)
		if meter is None:
			raise ValueError("no time signature in the 1st measure of the MusicXML")

		marks = list(measure.recurse().getElementsByClass(music21.expressions.RehearsalMark))
		if index == 0 or changed or marks:
			start = 0 if index == 0 else pickup + _eighths(measure, voice)
			if start.denominator != 1:
				raise ValueError(f"measure {measure.number} does not start on an 8th note")
			section = section_from_label(str(marks[0].content)) if marks else None
			starts.append((int(start), meter, section))

	return starts


def _tempos(score: music21.stream.Stream, pickup: Fraction) -> list[tuple[Fraction, float]]:
	"""Quarter-note tempo marks of `score` in time order, with where they take effect."""
	tempos = [
		(pickup + _eighths(mark, score), float(bpm))
		for mark in score.recurse().getElementsByClass(music21.tempo.MetronomeMark)
		if (bpm := mark.getQuarterBPM()) is not None and bpm > 0
	]
	return sorted(tempos, key=lambda tempo: tempo[0])


def _tempo_at(tempos: list[tuple[Fraction, float]], start: int, meter: Meter) -> int | None:
	in_effect = [bpm for at, bpm in tempos if at <= start] or [bpm for _, bpm in tempos[:1]]
	return beat_tempo_from_quarter(in_effect[-1], meter) if in_effect else None


def _chord_symbols(
	voice: music21.stream.Stream, pickup: Fraction
) -> list[tuple[int, music21.harmony.ChordSymbol]]:
	"""Chord symbols of `voice` in time order, each at the 8th note it starts on; the last of
	several on 1 8th note wins."""
	symbols: dict[int, music21.harmony.ChordSymbol] = {}
	for symbol in voice.recurse().getElementsByClass(music21.harmony.ChordSymbol):
		symbols[round(pickup + _eighths(symbol, voice))] = symbol

	return sorted(symbols.items(), key=lambda item: item[0])


def _part_chords(
	symbols: list[tuple[int, music21.harmony.ChordSymbol]], start: int, end: int, key: Key
) -> tuple[tuple[ChordEvent, ...] | None, list[int]]:
	"""Chord events from `start` to `end`, with where an inexact chord starts, or None when no
	chord symbol sounds there."""
	before = [symbol for at, symbol in symbols if at <= start]
	timeline: list[tuple[int, music21.harmony.ChordSymbol | None]] = [
		(start, before[-1] if before else None),
		*((at, symbol) for at, symbol in symbols if start < at < end),
	]
	if len(timeline) == 1 and timeline[0][1] is None:
		return None, []

	events: list[ChordEvent] = []
	inexact: list[int] = []
	for (at, symbol), (until, _) in zip(timeline, [*timeline[1:], (end, None)], strict=True):
		chord = None
		if symbol is not None and not isinstance(symbol, music21.harmony.NoChord):
			chord, exact = chord_from_symbol(symbol, key)
			if not exact:
				inexact.append(at)

		for piece in range(at, until, MAX_DURATION):
			events.append(ChordEvent(chord, min(MAX_DURATION, until - piece)))

	return tuple(events), inexact


def _has_repeats(score: music21.stream.Stream) -> bool:
	return bool(
		score.recurse().getElementsByClass(_REPEATS)
		or score.spannerBundle.getByClass(music21.spanner.RepeatBracket)
	)


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
