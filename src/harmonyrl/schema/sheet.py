"""A lead sheet: a melody in consecutive parts, each with chords or none yet."""

from dataclasses import dataclass

from .chord import ChordEvent
from .segment import Melody


@dataclass(frozen=True)
class SheetPart:
	"""A stretch of a lead sheet with 1 meter and section, and the chords that cover its melody,
	or None when it has none yet."""

	melody: Melody
	chords: tuple[ChordEvent, ...] | None = None

	def __post_init__(self):
		if self.melody.length % self.melody.meter.bar_eighths:
			raise ValueError(
				f"a part must last whole bars, got {self.melody.length} 8th notes "
				f"in bars of {self.melody.meter.bar_eighths}"
			)

		if self.chords is not None and sum(e.duration for e in self.chords) != self.melody.length:
			raise ValueError(
				f"chords must cover the melody's {self.melody.length} 8th notes exactly."
			)


@dataclass(frozen=True)
class LeadSheet:
	"""Parts of a melody in time order."""

	parts: tuple[SheetPart, ...]
	title: str | None = None

	def __post_init__(self) -> None:
		if not self.parts:
			raise ValueError("a lead sheet needs at least 1 part")
