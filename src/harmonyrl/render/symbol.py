"""Key signatures, enharmonic spellings, and lead-sheet chord symbols."""

import music21

from harmonyrl.schema import Key, Mode, RawChord, chord_pitch_classes

_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_NAMES = ("C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B")
_NEUTRAL_NAMES = ("C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")

_MAJOR_SHARPS = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)
_REL_MAJOR_OFFSET: dict[Mode, int] = {
	Mode.MAJOR: 0,
	Mode.DORIAN: 2,
	Mode.PHRYGIAN: 4,
	Mode.PHRYGIAN_DOMINANT: 4,
	Mode.LYDIAN: 5,
	Mode.MIXOLYDIAN: 7,
	Mode.MINOR: 9,
	Mode.HARMONIC_MINOR: 9,
	Mode.LOCRIAN: 11,
}

_ADD_STEP: dict[int, str] = {
	1: "addb9",
	2: "add9",
	3: "add#9",
	4: "add3",
	5: "add11",
	6: "add#11",
	7: "add5",
	8: "addb13",
	10: "add7",
	11: "add#7",
}

_KIND_REWRITES: dict[str, tuple[str, tuple[tuple[str, int, int], ...]]] = {
	"suspended-fourth-seventh": ("suspended-fourth", (("add", 7, 0),)),
	"augmented-major-seventh": ("major-seventh", (("alter", 5, 1),)),
	"minor-major-ninth": ("minor-major-seventh", (("add", 9, 0),)),
	"minor-major-11th": ("minor-major-seventh", (("add", 9, 0), ("add", 11, 0))),
}

_MODIFICATIONS: dict[str, tuple[str, int, int]] = {
	"addb9": ("add", 9, -1),
	"add9": ("add", 9, 0),
	"add#9": ("add", 9, 1),
	"add3": ("add", 3, 0),
	"add11": ("add", 11, 0),
	"add#11": ("add", 11, 1),
	"add5": ("add", 5, 0),
	"add6": ("add", 13, 0),
	"add13": ("add", 13, 0),
	"addb13": ("add", 13, -1),
	"add7": ("add", 7, 0),
	"add#7": ("add", 7, 1),
	"alterb5": ("alter", 5, -1),
	"alter#5": ("alter", 5, 1),
}


def key_sharps(key: Key) -> int:
	"""Number of sharps (negative for flats) in the key signature of `key`."""
	rel_major = (key.tonic - _REL_MAJOR_OFFSET[key.mode]) % 12
	return _MAJOR_SHARPS[rel_major]


def midi_key_number(key: Key) -> int:
	"""pretty_midi KeySignature number (0-11 major, 12-23 minor) matching `key`'s signature."""
	if key.mode in (Mode.MINOR, Mode.HARMONIC_MINOR):
		return key.tonic + 12
	return (key.tonic - _REL_MAJOR_OFFSET[key.mode]) % 12


def chord_symbol(chord: RawChord, key: Key) -> music21.harmony.ChordSymbol:
	"""ChordSymbol for `chord` in `key`, with root spelling from the active scale and
	quality from the root-relative interval structure.

	Subtraction degrees are never emitted: score software renders them as "no3"/"no5".
	A missing fifth is left unnotated, a missing third only shows through sus kinds or the
	power kind, and maj7-based sus chords (no MusicXML kind exists) carry their name in the
	kind text attribute over the unspecified kind — MuseScore drops the root of a
	"none"-kind harmony, and "other" keeps both."""
	bass_pc, pitch_classes = chord_pitch_classes(chord, key)
	names = pitch_names(_spelling_key(chord, key))
	root_pc = _root_pitch_class(chord, key)

	intervals = frozenset((pc - root_pc) % 12 for pc in pitch_classes)
	kind, mods = _suffix_plan(intervals)

	figure = names[root_pc]
	if bass_pc != root_pc:
		figure += "/" + names[bass_pc]

	if kind == "5":
		cs = music21.harmony.ChordSymbol(figure=figure)
		cs.chordKind = "power"
		_apply_modifications(cs, mods)
	elif kind in ("maj7sus2", "maj7sus4"):
		cs = music21.harmony.ChordSymbol(figure=figure)
		cs.chordKind = "other"
		cs.chordKindStr = kind
		_apply_modifications(cs, mods)
	elif kind == "7sus2":
		cs = music21.harmony.ChordSymbol(figure=figure + "sus2" + "".join(mods))
		_apply_modifications(cs, ["add7"])
	else:
		cs = music21.harmony.ChordSymbol(figure=figure + kind + "".join(mods))

	_rewrite_kind(cs)

	return cs


def _apply_modifications(cs: music21.harmony.ChordSymbol, mods: list[str]) -> None:
	for mod in mods:
		cs.addChordStepModification(
			music21.harmony.ChordStepModification(*_MODIFICATIONS[mod]),
			updatePitches=False,
		)


def _rewrite_kind(cs: music21.harmony.ChordSymbol) -> None:
	if cs.chordKind not in _KIND_REWRITES:
		return

	kind, degrees = _KIND_REWRITES[cs.chordKind]
	cs.chordKind = kind

	for mod_type, degree, alter in degrees:
		cs.addChordStepModification(
			music21.harmony.ChordStepModification(mod_type, degree, alter),
			updatePitches=False,
		)


def _spelling_key(chord: RawChord, key: Key) -> Key:
	base = Key(key.tonic, chord.borrowed) if isinstance(chord.borrowed, Mode) else key

	if not chord.applied:
		return base

	scale = chord.borrowed if isinstance(chord.borrowed, tuple) else base.mode.scale
	target_pc = (key.tonic + scale[chord.root - 1]) % 12

	return Key(target_pc, Mode.MAJOR)


def pitch_names(key: Key) -> dict[int, str]:
	"""Pitch-class to note-name spellings: the key's scale degrees first, then accidentals
	that follow the direction of its key signature (flats as '-', music21 style)."""
	sharps = key_sharps(key)
	fallback = _SHARP_NAMES if sharps > 0 else _FLAT_NAMES if sharps < 0 else _NEUTRAL_NAMES
	tonic_name = fallback[key.tonic]
	names: dict[int, str] = {}
	letter = _LETTERS.index(tonic_name[0])

	for offset in key.mode.scale:
		target = (key.tonic + offset) % 12
		natural = _LETTER_PC[_LETTERS[letter]]

		diff = (target - natural) % 12
		if diff > 6:
			diff -= 12
		if diff in (-1, 0, 1):
			names[target] = _LETTERS[letter] + {-1: "-", 0: "", 1: "#"}[diff]

		letter = (letter + 1) % 7

	return {pc: names.get(pc, fallback[pc]) for pc in range(12)}


def _root_pitch_class(chord: RawChord, key: Key) -> int:
	"""Absolute pitch class of `chord`'s root, mirroring the schema's stacking rules."""
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


def _suffix_plan(intervals: frozenset[int]) -> tuple[str, list[str]]:
	"""Kind token and modification strings for the root-relative interval structure.

	Omitted degrees are never expressed as subtraction mods: an absent fifth is not
	notated, and a missing third only shows through sus kinds or the power kind "5"."""
	has_maj3 = 4 in intervals
	has_min3 = not has_maj3 and 3 in intervals
	has_p5 = 7 in intervals
	has_dim5 = not has_p5 and 6 in intervals
	has_aug5 = not has_p5 and not has_dim5 and 8 in intervals
	has_maj7 = 11 in intervals
	has_min7 = not has_maj7 and 10 in intervals
	is_dim7 = has_min3 and has_dim5 and not has_maj7 and not has_min7 and 9 in intervals

	used = {0}
	mods: list[str] = []

	if is_dim7:
		kind = "dim7"
		used |= {3, 6, 9}
	elif has_min3 and has_dim5 and has_min7:
		kind = "m7b5"
		used |= {3, 6, 10}
	elif has_maj3 or has_min3:
		used.add(4 if has_maj3 else 3)
		if has_aug5 and has_maj3:
			used.add(8)
			if has_maj7:
				kind = "augmaj7"
				used.add(11)
			elif has_min7:
				kind = "aug7"
				used.add(10)
			else:
				kind = "aug"
		elif has_dim5 and has_min3 and not (has_maj7 or has_min7):
			kind = "dim"
			used.add(6)
		else:
			if has_p5:
				used.add(7)
			elif has_dim5:
				used.add(6)
				mods.append("alterb5")
			elif has_aug5:
				used.add(8)
				mods.append("alter#5")

			prefix = "" if has_maj3 else "m"
			if has_maj7 or has_min7:
				used.add(11 if has_maj7 else 10)
				kind, extra = _extended_seventh_kind(
					intervals - used, has_maj3=has_maj3, has_maj7=has_maj7, allow_stack=not mods
				)
				used |= extra
			elif not mods and 9 in intervals:
				kind = f"{prefix}6"
				used.add(9)
			else:
				kind = prefix
	else:
		suspension = 4 if 5 in intervals else 2 if 2 in intervals else 0
		if suspension:
			used.add(5 if suspension == 4 else 2)
		if has_maj7:
			used.add(11)
			kind = "maj7sus4" if suspension == 4 else "maj7sus2" if suspension == 2 else "maj7"
		elif has_min7:
			used.add(10)
			kind = "7sus4" if suspension == 4 else "7sus2" if suspension == 2 else "7"
		elif suspension:
			kind = "sus4" if suspension == 4 else "sus2"
		elif has_p5 or has_dim5 or has_aug5:
			kind = "5"
		else:
			kind = ""

		if has_p5:
			used.add(7)
		elif has_dim5:
			used.add(6)
			mods.append("alterb5")
		elif has_aug5:
			used.add(8)
			mods.append("alter#5")

	has_seventh = has_maj7 or has_min7 or is_dim7
	for step in sorted(intervals - used):
		if step == 9:
			mods.append("add13" if has_seventh else "add6")
		else:
			mods.append(_ADD_STEP[step])

	return kind, mods


def _extended_seventh_kind(
	remaining: frozenset[int], *, has_maj3: bool, has_maj7: bool, allow_stack: bool
) -> tuple[str, frozenset[int]]:
	base7 = ("maj7" if has_maj7 else "7") if has_maj3 else ("mM7" if has_maj7 else "m7")
	if not allow_stack or 10 in remaining:
		return base7, frozenset()

	can_9 = 2 in remaining and 1 not in remaining and not (has_maj3 and 3 in remaining)
	can_11 = can_9 and 5 in remaining and 6 not in remaining
	can_13 = can_11 and 9 in remaining and 8 not in remaining
	prefix = ("Maj" if has_maj7 else "") if has_maj3 else ("mM" if has_maj7 else "m")

	if can_13:
		return f"{prefix}13", frozenset({2, 5, 9})
	if can_11:
		return f"{prefix}11", frozenset({2, 5})
	if can_9:
		return f"{prefix}9", frozenset({2})
	return base7, frozenset()
