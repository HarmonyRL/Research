"""Model vocabulary: the chord attributes a model predicts, one categorical head each.

Chord heads are named after `RawChord` fields, so `RawChord(**values)` builds a predicted chord.
"""

from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations

from .chord import MAX_DURATION, RawChord
from .key import Key, Mode
from .pitch import chord_pitch_classes


def _subsets[T: (int, str)](items: tuple[T, ...]) -> tuple[tuple[T, ...], ...]:
	return tuple(
		tuple(sorted(subset))
		for size in range(len(items) + 1)
		for subset in combinations(items, size)
	)


HEAD_VALUES: dict[str, tuple[object, ...]] = {
	"root": (None, 1, 2, 3, 4, 5, 6, 7),
	"applied": tuple(range(8)),
	"borrowed": (None, *Mode),
	"type": (5, 7, 9, 11, 13),
	"suspensions": _subsets((2, 4)),
	"adds": _subsets((4, 6, 9)),
	"omits": _subsets((3, 5)),
	"alterations": _subsets(("b5", "#5", "b9", "#9", "#11", "b13")),
	"inversion": (0, 1, 2, 3),
	"duration": tuple(range(1, MAX_DURATION + 1)),
}
"""Output heads in prediction order and the values each can take. A root of None is no chord,
after which only the duration head is predicted."""


def allowed(
	head: str, prefix: Mapping[str, object], *, previous_no_chord: bool, remaining: int
) -> tuple[bool, ...]:
	"""Mask over `HEAD_VALUES[head]` given the heads already predicted for this event.

	No chord may not follow no chord, since consecutive no-chord events are 1 event. A triad
	has no 3rd inversion. A duration may not run past the `remaining` 8th notes of the segment.
	"""
	if remaining < 1:
		raise ValueError(f"remaining must be positive, got {remaining}")

	values = HEAD_VALUES[head]
	if head == "root":
		return tuple(not (previous_no_chord and value is None) for value in values)
	if head == "inversion":
		max_inversion = 2 if prefix["type"] == 5 else 3
		return tuple(isinstance(value, int) and value <= max_inversion for value in values)
	if head == "duration":
		return tuple(isinstance(value, int) and value <= remaining for value in values)

	return (True,) * len(values)


def project(chord: RawChord, key: Key) -> RawChord:
	"""Nearest chord the vocabulary can represent.

	Only a borrowed scale given as offsets is not representable. It becomes the mode,
	or no borrowing, whose chord shraes the most pitch classes with the original.
	"""
	if not isinstance(chord.borrowed, tuple):
		return chord

	target = chord_pitch_classes(chord, key)[1]
	candidates = [replace(chord, borrowed=mode) for mode in (None, *Mode)]

	return max(candidates, key=lambda c: _jaccard(chord_pitch_classes(c, key)[1], target))


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
	return len(a & b) / len(a | b)
