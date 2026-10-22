"""Lead-sheet scores: a melody staff with chord symbols, meter, and key signatures."""

import re
from collections.abc import Sequence
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import music21

from harmonyrl.schema import ChordEvent, Key, Melody, Meter, Mode, chord_pitch_classes

from .voicing import check_tiling

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_MAJOR_TONIC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_TONIC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")


def melody_to_score(
	melody: Melody, events: Sequence[ChordEvent], *, tempo: int | None = None
) -> music21.stream.Score:
	"""A lead sheet of a melody with chord events: melody with chord symbols above
	the staff.

	`tempo`, when known, becomes a metronome mark so the printed score carries the
	playback tempo of the audio rendering.

	Measures are built explicitly with notes split at barlines, rests cut at metric
	boundaries, and symbols placed inline. music21's automatic measure assembly
	mangles exactly these (backup voice layouts, meter-blind rest splitting), so it
	is bypassed rather than fought.
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
	if melody.notes:
		elements.extend(_rest_events(previous_end, Fraction(melody.length), melody.meter))

	position = Fraction(0)
	for event in events:
		if event.chord is not None:
			elements.append((position, _chord_symbol(event.chord, melody.key_at(position))))
		position += event.duration

	signatures: dict[Fraction, music21.key.KeySignature] = {}
	previous: Key | None = None
	for span in melody.keys:
		if span.key != previous:
			signature = _key_signature(span.key)
			if signature is not None:
				signatures[Fraction(span.start)] = signature
			previous = span.key

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
				measure.insert(0, music21.tempo.MetronomeMark(number=tempo))
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


def _note_events(note, bar: int) -> list[tuple[Fraction, music21.note.Note]]:
	"""One melody note as export events, split at barlines into tied pieces."""
	spans: list[tuple[Fraction, Fraction]] = []
	onset = note.onset
	remaining = note.duration
	while remaining > 0:
		to_boundary = bar - (onset % bar) if onset % bar else Fraction(bar)
		length = min(remaining, to_boundary)
		spans.append((onset, length))
		onset += length
		remaining -= length
	rendered = []
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


_DOTTABLE = (128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
"""Rest lengths in 8th notes, longest first, that one dotted rest can express."""


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


def _key_signature(key: Key) -> music21.key.KeySignature | None:
	if key.mode == Mode.MINOR:
		name = _MINOR_TONIC_NAMES[key.tonic]
	else:
		name = _MAJOR_TONIC_NAMES[key.tonic]
	try:
		return music21.key.KeySignature(music21.key.Key(name, key.mode.value).sharps)
	except music21.exceptions21.Music21Exception:
		return None


def _spellings(key: Key) -> dict[int, str]:
	"""Pitch-class to note-name spellings: the key's own scale, then accidentals that
	follow the direction of its key signature."""
	tonic_name = (
		_MINOR_TONIC_NAMES[key.tonic] if key.mode == Mode.MINOR else _MAJOR_TONIC_NAMES[key.tonic]
	)
	tonic_pc = key.tonic
	names: dict[int, str] = {}
	letter = _LETTERS.index(tonic_name[0])
	for offset in key.mode.scale:
		target = (tonic_pc + offset) % 12
		natural = _LETTER_PC[_LETTERS[letter]]
		diff = (target - natural) % 12
		if diff > 6:
			diff -= 12
		if diff not in (-2, -1, 0, 1, 2):
			continue
		names[target] = _LETTERS[letter] + {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}[diff]
		letter = (letter + 1) % 7
	fallback = (
		_SHARP_NAMES if (music21.key.Key(tonic_name, "major").sharps or 0) >= 0 else _FLAT_NAMES
	)
	return {pc: names.get(pc, fallback[pc]) for pc in range(12)}


def _root_pitch_class(chord, key: Key) -> int:
	"""Absolute pitch class of a chord's root, mirroring the schema's stacking rules."""
	scale = (
		chord.borrowed
		if isinstance(chord.borrowed, tuple)
		else key.mode.scale
		if chord.borrowed is None
		else chord.borrowed.scale
	)
	root_offset = scale[chord.root - 1]
	if chord.applied:
		root_offset += Mode.MAJOR.scale[chord.applied - 1]
	return (key.tonic + root_offset) % 12


def _chord_symbol(chord, key: Key) -> music21.base.Music21Object:
	"""Chord symbol for one event chord: root spelling from the key, quality from the
	root-relative interval structure. Unparseable figures degrade to visible text."""
	bass_pc, pitch_classes = chord_pitch_classes(chord, key)
	names = _spellings(key)
	root_pc = _root_pitch_class(chord, key)
	intervals = frozenset((pc - root_pc) % 12 for pc in pitch_classes)
	figure = _figure_name(names[root_pc]) + _figure_suffix(intervals)
	if bass_pc != root_pc:
		figure += "/" + _figure_name(names[bass_pc])
	try:
		return music21.harmony.ChordSymbol(figure=figure)
	except Exception:
		return music21.expressions.TextExpression(figure)


def _figure_name(name: str) -> str:
	"""music21 chord figures spell flats with '-', unlike the 'b' of note names."""
	return name[0] + name[1:].replace("b", "-")


def _figure_suffix(intervals: frozenset[int]) -> str:
	third_minor, third_major = 3 in intervals, 4 in intervals
	fifth_dim, fifth_perfect, fifth_aug = 6 in intervals, 7 in intervals, 8 in intervals
	seventh_dim, seventh_minor, seventh_major = 9 in intervals, 10 in intervals, 11 in intervals
	has_seventh = seventh_dim or seventh_minor or seventh_major

	base = None
	if third_major and not fifth_dim:
		base = "aug" if fifth_aug and not fifth_perfect else ""
	elif third_minor and not fifth_aug:
		if fifth_dim and seventh_minor:
			base = "m7b5"
		elif fifth_dim and seventh_dim:
			base = "dim7"
		elif fifth_dim:
			base = "dim"
		else:
			base = "m"
	elif not third_minor and not third_major:
		base = ("sus2" if 2 in intervals else "") + ("sus4" if 5 in intervals else "")

	if base is None:
		return "[" + ",".join(map(str, sorted(intervals))) + "]"

	seventh = ""
	if seventh_minor:
		seventh = "7"
	elif seventh_major and base in ("", "m", "aug"):
		seventh = "maj7"

	tensions = []
	if 1 in intervals:
		tensions.append("b9")
	if 2 in intervals:
		tensions.append("9" if has_seventh else "add9")
	if 3 in intervals and third_major:
		tensions.append("#9")
	if 5 in intervals and (third_minor or third_major):
		tensions.append("11")
	if 6 in intervals and fifth_perfect:
		tensions.append("#11")
	if 8 in intervals and fifth_perfect:
		tensions.append("b13")
	if 9 in intervals:
		tensions.append("13" if has_seventh else "6")
	return base + seventh + "".join(tensions)
