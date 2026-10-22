"""Modes and keys. Scales are 7 semitone offsets from the tonic."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import accumulate


class Mode(StrEnum):
	"""Scale of a key or of a borrowed chord. Values are the Hooktheory mode names."""

	MAJOR = "major"
	MINOR = "minor"
	DORIAN = "dorian"
	PHRYGIAN = "phrygian"
	LYDIAN = "lydian"
	MIXOLYDIAN = "mixolydian"
	LOCRIAN = "locrian"
	HARMONIC_MINOR = "harmonicMinor"
	PHRYGIAN_DOMINANT = "phrygianDominant"

	@property
	def scale(self) -> tuple[int, ...]:
		"""7 semitone offsets from the tonic."""
		return _SCALES[self]


_SCALES: dict[Mode, tuple[int, ...]] = {
	Mode.MAJOR: (0, 2, 4, 5, 7, 9, 11),
	Mode.MINOR: (0, 2, 3, 5, 7, 8, 10),
	Mode.DORIAN: (0, 2, 3, 5, 7, 9, 10),
	Mode.PHRYGIAN: (0, 1, 3, 5, 7, 8, 10),
	Mode.LYDIAN: (0, 2, 4, 6, 7, 9, 11),
	Mode.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
	Mode.LOCRIAN: (0, 1, 3, 5, 6, 8, 10),
	Mode.HARMONIC_MINOR: (0, 2, 3, 5, 7, 8, 11),
	Mode.PHRYGIAN_DOMINANT: (0, 1, 4, 5, 7, 8, 10),
}


def mode_from_intervals(intervals: Sequence[int]) -> Mode:
	"""Mode whose scale has the given 6 steps between consecutive degrees."""
	scale = (0, *accumulate(intervals))
	for mode in Mode:
		if mode.scale == scale:
			return mode

	raise ValueError(f"No mode has scale steps {list(intervals)}")


@dataclass(frozen=True)
class Key:
	"""A tonic pitch class (0 = C) and a mode."""

	tonic: int
	mode: Mode

	def __post_init__(self) -> None:
		if not 0 <= self.tonic < 12:
			raise ValueError(f"tonic must be a pitch class in 0..11, got {self.tonic}")


@dataclass(frozen=True)
class KeySpan:
	"""A key that holds from `start` (in 8th notes) until the next span begins."""

	start: int
	key: Key

	def __post_init__(self) -> None:
		if self.start < 0:
			raise ValueError(f"start must be non-negative, got {self.start}")
