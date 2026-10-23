"""Lead-sheet scores: a melody staff with chord symbols, meter, and key signatures."""

import re
from collections.abc import Sequence
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import music21

from harmonyrl.schema import ChordEvent, Key, Melody, Meter, Note

from .symbol import chord_symbol, key_sharps
from .voicing import check_tiling

_DOTTABLE = (128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
"""Rest lengths in 8th notes, longest first, that one dotted rest can express."""


def melody_to_score(
	melody: Melody, events: Sequence[ChordEvent], *, tempo: int | None = None
) -> music21.stream.Score:
	"""A lead sheet of a melody with chord events: melody with chord symbols above
	the staff.

	`tempo`, when known, becomes a metronome mark (quarter note for x/4 meters,
	dotted quarter for x/8 meters) so the printed score carries the playback tempo
	of the audio rendering.

	Measures are built explicitly with notes split at barlines and into single-notehead
	components, rests cut at metric boundaries, and symbols placed inline. music21's
	automatic measure assembly mangles these (backup voice layouts, meter-blind rest
	splitting), so it is bypassed rather than fought.
	"""
	check_tiling(melody, events)
	bar = melody.meter.bar_eighths
	part = music21.stream.Part()
	part.partName = "Melody"

	elements: list[tuple[Fraction, music21.base.Music21Object]] = []
	previous_end = Fraction(0)
	for note in melody.notes:
		if note.onset > previous_end:
			elements.extend(_rest_events(previous_end, note.onset, melody.meter))
		elements.extend(_note_events(note, bar))
		previous_end = note.offset
	if previous_end < melody.length:
		elements.extend(_rest_events(previous_end, Fraction(melody.length), melody.meter))

	position = Fraction(0)
	for event in events:
		if event.chord is not None:
			elements.append((position, chord_symbol(event.chord, melody.key_at(position))))
		position += event.duration

	signatures: dict[Fraction, music21.key.KeySignature] = {}
	previous_key: Key | None = None
	for span in melody.keys:
		if span.key != previous_key:
			signatures[Fraction(span.start)] = music21.key.KeySignature(key_sharps(span.key))
			previous_key = span.key

	elements.sort(key=lambda element: element[0])
	measure_count = ceil(Fraction(melody.length, bar))
	cursor = 0
	for index in range(measure_count):
		measure = music21.stream.Measure(number=index + 1)
		bar_start = index * bar
		if index == 0:
			measure.insert(
				0,
				music21.meter.TimeSignature(
					f"{melody.meter.beats_per_bar}/{melody.meter.beat_unit}"
				),
			)
			if tempo is not None:
				referent = music21.duration.Duration(1.5 if melody.meter.beat_unit == 8 else 1.0)
				measure.insert(0, music21.tempo.MetronomeMark(number=tempo, referent=referent))
		for start, signature in signatures.items():
			if bar_start <= start < bar_start + bar:
				measure.insert(Fraction(start - bar_start, 2), signature)
		while cursor < len(elements) and elements[cursor][0] < bar_start + bar:
			start, element = elements[cursor]
			measure.insert(Fraction(start - bar_start, 2), element)
			cursor += 1
		measure.duration = music21.duration.Duration(Fraction(bar, 2))
		part.append(measure)

	score = music21.stream.Score()
	score.insert(0, part)
	return score


def _note_events(note: Note, bar: int) -> list[tuple[Fraction, music21.note.Note]]:
	"""One melody note as export events, split at barlines and into single-notehead
	pieces connected with ties."""
	spans: list[tuple[Fraction, Fraction]] = []
	onset = note.onset
	remaining = note.duration
	while remaining > 0:
		to_boundary = bar - (onset % bar) if onset % bar else Fraction(bar)
		bar_piece = min(remaining, to_boundary)
		for length in _notehead_lengths(bar_piece):
			spans.append((onset, length))
			onset += length
		remaining -= bar_piece

	rendered: list[tuple[Fraction, music21.note.Note]] = []
	for i, (start, length) in enumerate(spans):
		piece = music21.note.Note()
		piece.pitch.midi = note.pitch
		piece.quarterLength = Fraction(length, 2)
		if len(spans) > 1:
			piece.tie = music21.tie.Tie(
				"start" if i == 0 else "stop" if i == len(spans) - 1 else "continue"
			)
		rendered.append((start, piece))
	return rendered


def _notehead_lengths(length: Fraction) -> list[Fraction]:
	duration = music21.duration.Duration(Fraction(length, 2))
	if len(duration.components) <= 1:
		return [length]
	return [Fraction(component.quarterLength) * 2 for component in duration.components]


def _rest_events(
	start: Fraction, end: Fraction, meter: Meter
) -> list[tuple[Fraction, music21.note.Rest]]:
	"""Rests filling [start, end), cut at barlines and strong beats.

	Cutting at metric boundaries leaves pieces a single well-shaped rest can express;
	music21's automatic gap filling splits without regard for the meter and renders
	as tied or misaligned rests in meters like 6/4.
	"""
	bar = meter.bar_eighths
	strong = {beat * meter.eighths_per_beat for beat in meter.strong_beats()} | {0}
	cuts = sorted(
		{position for position in range(floor(start) + 1, ceil(end)) if position % bar in strong}
	)
	pieces = []
	cursor = start
	for cut in cuts:
		pieces.append((cursor, cut - cursor))
		cursor = Fraction(cut)
	pieces.append((cursor, end - cursor))
	events: list[tuple[Fraction, music21.note.Rest]] = []
	for offset, piece in pieces:
		lengths: list[Fraction] = [piece]
		if piece.denominator == 1 and int(piece) not in _DOTTABLE:
			lengths = [Fraction(n) for n in _split_dottable(int(piece))]
		for length in lengths:
			rest = music21.note.Rest()
			rest.quarterLength = Fraction(length, 2)
			events.append((offset, rest))
			offset += length
	return events


def _split_dottable(length: int) -> list[int]:
	pieces = []
	while length:
		piece = next(size for size in _DOTTABLE if size <= length)
		pieces.append(piece)
		length -= piece
	return pieces


def write_score(score: music21.stream.Score, path: Path) -> None:
	"""Write MusicXML deterministically, fixing what music21 emits wrongly for a
	single-voice score: random part ids, the generation date, and voice number 0,
	which the MusicXML spec and MuseScore reject."""
	score.write("musicxml", fp=str(path))
	_normalize(path)


def _normalize(path: Path) -> None:
	text = path.read_text(encoding="utf-8")
	text = re.sub(r"<encoding-date>.*?</encoding-date>\n?", "", text, flags=re.DOTALL)
	text = re.sub(r"P[0-9a-f]{32}", "P1", text)
	text = text.replace("<voice>0</voice>", "<voice>1</voice>")
	path.write_text(text, encoding="utf-8")
