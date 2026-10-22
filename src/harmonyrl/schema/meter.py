"""Meters: bar length, strong beats, and beat strength in 8-note units."""

import logging
from dataclasses import dataclass
from functools import cache

_logger = logging.getLogger(__name__)

_STRONG_BEATS: dict[tuple[int, int], tuple[int, ...]] = {
	(2, 4): (0,),
	(3, 4): (0,),
	(3, 8): (0,),
	(4, 4): (0, 2),
	(5, 4): (0, 3),  # 3+2 grouping assumed
	(6, 4): (0, 3),
	(6, 8): (0, 3),
	(8, 4): (0, 4),
	(9, 4): (0, 3, 6),
	(9, 8): (0, 3, 6),
	(12, 4): (0, 3, 6, 9),
	(12, 8): (0, 3, 6, 9),
}


@dataclass(frozen=True)
class Meter:
	"""A time signature. `beat_unit` is the denominator and must be 4 or 8."""

	beats_per_bar: int
	beat_unit: int

	def __post_init__(self) -> None:
		if self.beats_per_bar < 1:
			raise ValueError(f"beats_per_bar must be positive, got {self.beats_per_bar}")

		if self.beat_unit not in (4, 8):
			raise ValueError(f"beat_unit must be 4 or 8, got {self.beat_unit}")

	@property
	def eighths_per_beat(self) -> int:
		"""Number of 8th notes in one beat."""
		return 8 // self.beat_unit

	@property
	def bar_eighths(self) -> int:
		"""Number of 8th notes in one bar."""
		return self.beats_per_bar * self.eighths_per_beat

	def strong_beats(self) -> tuple[int, ...]:
		"""Strong beats as 0-based beat indices within a bar."""
		key = (self.beats_per_bar, self.beat_unit)
		if key not in _STRONG_BEATS:
			_warn_untabulated(*key)
			return (0,)

		return _STRONG_BEATS[key]

	def beat_strength(self, position: int) -> int:
		"""Metric weight of an 8th-note position counted from a barline.

		3 is the downbeat, 2 another strong beat, 1 another beat onset, and 0 off the beat.
		"""
		beat, offset = divmod(position % self.bar_eighths, self.eighths_per_beat)
		if offset:
			return 0
		if beat == 0:
			return 3
		return 2 if beat in self.strong_beats() else 1


@cache
def _warn_untabulated(beats_per_bar: int, beat_unit: int) -> None:
	_logger.warning(
		"No strong-beat entry for %d/%d; using the downbeat only", beats_per_bar, beat_unit
	)
