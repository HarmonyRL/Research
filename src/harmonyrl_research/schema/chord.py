"""Chords as annotated in Hooktheory, and chord events measured in 8th notes."""

from dataclasses import dataclass

from .key import Mode

MAX_DURATION = 128
"""Longest chord event in 8th notes, the longest chord in the dataset."""

_TYPES = frozenset({5, 7, 9, 11, 13})
_ADDS = frozenset({4, 6, 9})
_OMITS = frozenset({3, 5})
_ALTERATIONS = frozenset({"b5", "#5", "b9", "#9", "#11", "b13"})
_SUSPENSIONS = frozenset({2, 4})


def _check_subset[T: (int, str)](name: str, values: tuple[T, ...], allowed: frozenset[T]) -> None:
	if not set(values) <= allowed:
		raise ValueError(f"{name} must be drawn from {sorted(allowed)}, got {list(values)}")


@dataclass(frozen=True)
class RawChord:
	"""A chord with every Hooktheory attribute, relative to the local key.

	`root` is a scale degree 1-7. For an applied chord, `applied` is its function degree
	(5 for V/, 7 for vii°/) and `root` is the degree it targets. `borrowed` is a mode or
	7 semitone offsets from the tonic. List attributes are stored sorted and deduplicated.
	"""

	root: int
	type: int
	inversion: int = 0
	applied: int = 0
	borrowed: Mode | tuple[int, ...] | None = None
	adds: tuple[int, ...] = ()
	omits: tuple[int, ...] = ()
	alterations: tuple[str, ...] = ()
	suspensions: tuple[int, ...] = ()

	def __post_init__(self) -> None:
		if not 1 <= self.root <= 7:
			raise ValueError(f"root must be a scale degree 1-7, got {self.root}")

		if self.type not in _TYPES:
			raise ValueError(f"type must be one of {sorted(_TYPES)}, got {self.type}")

		max_inversion = 2 if self.type == 5 else 3
		if not 0 <= self.inversion <= max_inversion:
			raise ValueError(
				f"inversion must be 0-{max_inversion} for type {self.type}, got {self.inversion}"
			)

		if not 0 <= self.applied <= 7:
			raise ValueError(f"applied must be 0 or a scale degree 1-7, got {self.applied}")

		if isinstance(self.borrowed, tuple) and len(self.borrowed) != 7:
			raise ValueError(f"borrowed scale must have 7 offsets, got {self.borrowed}")

		_check_subset("adds", self.adds, _ADDS)
		_check_subset("omits", self.omits, _OMITS)
		_check_subset("alterations", self.alterations, _ALTERATIONS)
		_check_subset("suspensions", self.suspensions, _SUSPENSIONS)

		for name in ("adds", "omits", "alterations", "suspensions"):
			object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))


@dataclass(frozen=True)
class ChordEvent:
	"""A chord held for `duration` 8th notes. `chord` is None when no chord sounds."""

	chord: RawChord | None
	duration: int

	def __post_init__(self) -> None:
		if not 1 <= self.duration <= MAX_DURATION:
			raise ValueError(f"duration must be 1-{MAX_DURATION}, got {self.duration}")
