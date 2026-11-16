"""Chord symbols of a score turned into the chord vocabulary of the policy."""

from functools import cache
from itertools import product
from typing import cast

import music21

from harmonyrl.schema import HEAD_VALUES, Key, Mode, RawChord, chord_pitch_classes

type _Sound = tuple[int, frozenset[int]]
"""Bass and pitch classes relative to the root."""

type _Function = tuple[int, int, Mode | None]
"""Root degree, applied degree, and borrowed mode."""

type _Quality = tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...], int]
"""Type, suspensions, adds, omits, alterations, and inversion."""

MAX_EXTRAS = 3
"""Most suspended, added, omitted, and altered tones a chord read from a score may have."""

_APPLIED = cast(tuple[int, ...], HEAD_VALUES["applied"])
_BORROWED = cast(tuple[Mode | None, ...], HEAD_VALUES["borrowed"])
_TYPES = cast(tuple[int, ...], HEAD_VALUES["type"])
_SUSPENSIONS = cast(tuple[tuple[int, ...], ...], HEAD_VALUES["suspensions"])
_ADDS = cast(tuple[tuple[int, ...], ...], HEAD_VALUES["adds"])
_OMITS = cast(tuple[tuple[int, ...], ...], HEAD_VALUES["omits"])
_ALTERATIONS = cast(tuple[tuple[str, ...], ...], HEAD_VALUES["alterations"])
_INVERSIONS = cast(tuple[int, ...], HEAD_VALUES["inversion"])


def chord_from_symbol(symbol: music21.harmony.ChordSymbol, key: Key) -> tuple[RawChord, bool]:
	"""The chord of the vocabulary that sounds `symbol` in `key`, and whether it sounds it
	exactly."""
	pitches = symbol.pitches
	if not pitches:
		raise ValueError(f"chord symbol {symbol.figure!r} has no pitches")

	root = symbol.root().pitchClass
	target: _Sound = (
		(symbol.bass().pitchClass - root) % 12,
		frozenset((pitch.pitchClass - root) % 12 for pitch in pitches),
	)
	functions = _functions(key.mode, (root - key.tonic) % 12)
	if not functions:
		raise ValueError(f"no chord of the vocabulary has the root of {symbol.figure!r}")

	for extras in range(MAX_EXTRAS + 1):
		for function in functions:
			chord = _sounds(key.mode, function, extras).get(target)
			if chord is not None:
				return chord, True

	def closeness(item: tuple[_Sound, RawChord]) -> tuple[bool, float, int]:
		(bass, pitch_classes), chord = item
		shared = len(pitch_classes & target[1]) / len(pitch_classes | target[1])
		return bass == target[0], shared, -_extras(chord)

	candidates = (
		item
		for extras in range(MAX_EXTRAS + 1)
		for function in functions
		for item in _sounds(key.mode, function, extras).items()
	)
	return max(candidates, key=closeness)[1], False


@cache
def _functions(mode: Mode, root: int) -> tuple[_Function, ...]:
	"""Functions whose root is `root` semitones above the tonic, plainest first."""
	found: list[_Function] = []
	for degree, applied, borrowed in product(range(1, 8), _APPLIED, _BORROWED):
		try:
			chord = RawChord(root=degree, type=5, applied=applied, borrowed=borrowed)
		except ValueError:
			continue

		if chord_pitch_classes(chord, Key(0, mode))[0] == root:
			found.append((degree, applied, borrowed))

	return tuple(sorted(found, key=_rank))


def _rank(function: _Function) -> tuple[int, int]:
	_, applied, borrowed = function

	if borrowed is None and applied in (0, 5, 7):
		return (0, (0, 5, 7).index(applied))
	if applied == 0 and borrowed in (Mode.MAJOR, Mode.MINOR):
		return (1, 0)
	if applied == 0:
		return (2, 0)
	return (3, applied)


@cache
def _qualities(extras: int) -> tuple[_Quality, ...]:
	"""Qualities with `extras` suspended, added, omitted, and altered tones, plainest first."""
	found = [
		quality
		for quality in product(_TYPES, _SUSPENSIONS, _ADDS, _OMITS, _ALTERATIONS, _INVERSIONS)
		if sum(len(part) for part in quality[1:5]) == extras
	]
	return tuple(sorted(found, key=lambda quality: (quality[0], quality[5])))


@cache
def _sounds(mode: Mode, function: _Function, extras: int) -> dict[_Sound, RawChord]:
	"""Every sound of `degree` with this function, mapped to its simplest chord."""
	degree, applied, borrowed = function
	key = Key(0, mode)
	root = chord_pitch_classes(
		RawChord(root=degree, type=5, applied=applied, borrowed=borrowed), key
	)[0]
	sounds: dict[_Sound, RawChord] = {}

	for type_, suspensions, adds, omits, alterations, inversion in _qualities(extras):
		try:
			chord = RawChord(
				root=degree,
				type=type_,
				inversion=inversion,
				applied=applied,
				borrowed=borrowed,
				adds=adds,
				omits=omits,
				alterations=alterations,
				suspensions=suspensions,
			)
		except ValueError:
			continue

		bass, pitch_classes = chord_pitch_classes(chord, key)
		sounds.setdefault(
			((bass - root) % 12, frozenset((pc - root) % 12 for pc in pitch_classes)), chord
		)

	return sounds


def _extras(chord: RawChord) -> int:
	return len(chord.suspensions) + len(chord.adds) + len(chord.omits) + len(chord.alterations)
