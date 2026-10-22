"""Lead-sheet scores: a melody staff with chord symbols, meter, and key signatures."""

import re
from fractions import Fraction
from pathlib import Path

import music21

from harmonyrl.schema import Key, Mode, Segment, chord_pitch_classes

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_MAJOR_TONIC_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_TONIC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")


def segment_to_score(segment: Segment) -> music21.stream.Score:
	"""A lead sheet of `segment`: melody with chord symbols above the staff."""
	part = music21.stream.Part()
	part.partName = "Melody"
	melody = segment.melody
	part.insert(
		0, music21.meter.TimeSignature(f"{melody.meter.beats_per_bar}/{melody.meter.beat_unit}")
	)

	previous: Key | None = None
	for span in melody.keys:
		if span.key != previous:
			signature = _key_signature(span.key)
			if signature is not None:
				part.insert(Fraction(span.start, 2), signature)
			previous = span.key

	for note in melody.notes:
		rendered = music21.note.Note()
		rendered.pitch.midi = note.pitch
		rendered.quarterLength = Fraction(note.duration, 2)
		part.insert(Fraction(note.onset, 2), rendered)
	if melody.notes and melody.notes[-1].offset < melody.length:
		rest = music21.note.Rest()
		rest.quarterLength = Fraction(melody.length - melody.notes[-1].offset, 2)
		part.insert(Fraction(melody.notes[-1].offset, 2), rest)

	position = Fraction(0)
	for event in segment.events:
		if event.chord is not None:
			part.insert(Fraction(position, 2), _chord_symbol(event.chord, melody.key_at(position)))
		position += event.duration

	score = music21.stream.Score()
	score.insert(0, part)
	return score


def write_score(score: music21.stream.Score, path: Path) -> None:
	"""Write MusicXML, dropping the generation date and random part ids so identical
	input gives identical bytes."""
	score.write("musicxml", fp=str(path))
	text = path.read_text(encoding="utf-8")
	text = re.sub(r"<encoding-date>.*?</encoding-date>\n?", "", text, flags=re.DOTALL)
	text = re.sub(r"P[0-9a-f]{32}", "P1", text)
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
