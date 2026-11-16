"""Lead-sheet scores: a melody staff with chord symbols, meter, and key signatures."""

import re
from bisect import bisect_right
from collections.abc import Sequence
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import music21

from harmonyrl.schema import ChordEvent, Key, Melody, Meter, Note

from .symbol import chord_symbol, key_sharps, pitch_names
from .voicing import check_tiling

DEFAULT_TITLE = "HarmonyRL Result"
"""Title of a score written without one."""


def write_score(score: music21.stream.Score, path: Path) -> None:
	"""Write MusicXML deterministically."""
	if score.metadata is None or not score.metadata.title:
		score.metadata = music21.metadata.Metadata(title=DEFAULT_TITLE)
	score.write("musicxml", fp=str(path), makeNotation=False)
	_normalize(path)


def melody_to_score(melody: Melody, events: Sequence[ChordEvent] | None) -> music21.stream.Score:
	"""A lead sheet of a melody with chord events: melody with chord symbols above the staff."""
	if events is not None:
		check_tiling(melody, events)

	bar = melody.meter.bar_eighths
	part = music21.stream.Part()
	part.partName = "Melody"

	changes = _key_changes(melody, bar)
	starts = sorted(changes)

	elements: list[tuple[Fraction, music21.base.Music21Object]] = []
	previous_end = Fraction(0)

	for note in melody.notes:
		if note.onset > previous_end:
			elements.extend(_rest_events(previous_end, note.onset, melody.meter))
		elements.extend(
			_note_events(note, bar, pitch_names(_display_key(note.onset, starts, changes)))
		)
		previous_end = note.offset

	measure_count = ceil(Fraction(melody.length, bar))
	if previous_end < measure_count * bar:
		elements.extend(_rest_events(previous_end, Fraction(measure_count * bar), melody.meter))

	symbols: list[tuple[Fraction, music21.harmony.Harmony]] = []
	position = Fraction(0)
	for event in events or ():
		if event.chord is not None:
			symbols.append((position, chord_symbol(event.chord, melody.key_at(position))))
		else:
			symbols.append((position, music21.harmony.NoChord()))

		position += event.duration

	elements.sort(key=lambda element: element[0])
	cursor = 0
	measures: list[music21.stream.Measure] = []

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
			measure.insert(0, music21.clef.TrebleClef())

			if melody.tempo is not None:
				referent = music21.duration.Duration(1.5 if melody.meter.beat_unit == 8 else 1.0)
				measure.insert(
					0, music21.tempo.MetronomeMark(number=melody.tempo, referent=referent)
				)

		if bar_start in changes:
			measure.insert(0, music21.key.KeySignature(key_sharps(changes[bar_start])))

		while cursor < len(elements) and elements[cursor][0] < bar_start + bar:
			start, element = elements[cursor]
			measure.insert(Fraction(start - bar_start, 2), element)
			cursor += 1

		measure.duration = music21.duration.Duration(Fraction(bar, 2))
		part.append(measure)
		measures.append(measure)

	for start in starts:
		if start > 0:
			measures[start // bar - 1].rightBarline = music21.bar.Barline("double")
	measures[-1].rightBarline = music21.bar.Barline("final")

	part.makeNotation(inPlace=True)
	for start, symbol in symbols:
		measures[int(start) // bar].insert(Fraction(int(start) % bar, 2), symbol)

	score = music21.stream.Score()
	score.insert(0, part)
	return score


def _key_changes(melody: Melody, bar: int) -> dict[int, Key]:
	"""Keys whose signature is drawn, keyed by the barline they are drawn at."""
	changes: dict[int, Key] = {}
	previous: Key | None = None

	for span in melody.keys:
		if span.key == previous:
			continue

		start = span.start if span.start % bar == 0 else (span.start // bar + 1) * bar
		if start >= melody.length and changes:
			break

		changes[start] = span.key
		previous = span.key

	return changes


def _display_key(onset: Fraction, starts: list[int], changes: dict[int, Key]) -> Key:
	"""The key whose signature is in effect at `onset`; spans snap to barlines."""
	return changes[starts[bisect_right(starts, onset) - 1]]


def _note_events(
	note: Note, bar: int, names: dict[int, str]
) -> list[tuple[Fraction, music21.note.Note]]:
	"""One melody note as export events, split at barlines and into single-notehead
	pieces connected with ties. Pitches are spelled by the active key so the key signature,
	not per-note accidentals, carries the diatonic spelling."""
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
	spelling = f"{names[note.pitch % 12]}{note.pitch // 12 - 1}"

	for i, (start, length) in enumerate(spans):
		piece = music21.note.Note(spelling)
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
	"""Rests filling [start, end), cut at barlines and strong beats."""
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
		for length in _notehead_lengths(piece):
			rest = music21.note.Rest()
			rest.quarterLength = Fraction(length, 2)
			events.append((offset, rest))
			offset += length

	return events


def _normalize(path: Path) -> None:
	text = path.read_text(encoding="utf-8")
	text = re.sub(r"<encoding-date>.*?</encoding-date>\n?", "", text, flags=re.DOTALL)
	text = re.sub(r"P[0-9a-f]{32}", "P1", text)
	text = text.replace("<voice>0</voice>", "<voice>1</voice>")
	# music21 credits itself as the composer of a score that names none.
	text = re.sub(r"\s*<creator type=\"composer\">Music21</creator>", "", text)
	path.write_text(text, encoding="utf-8")
