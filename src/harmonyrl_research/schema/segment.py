"""Melodies and annotated segments, timed in 8th notes from a barline at position 0."""

import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise

from .chord import MAX_DURATION, ChordEvent
from .key import Key, KeySpan
from .meter import Meter


@dataclass(frozen=True)
class Note:
	"""A melody note with times in 8th notes and a MIDI pitch.

	Times are exact fractions, so tuplets keep their positions. Floats are rejected; round source
	values to a fraction first.
	"""

	onset: Fraction
	duration: Fraction
	pitch: int

	def __post_init__(self) -> None:
		for name in ("onset", "duration"):
			value = getattr(self, name)
			if not isinstance(value, int | Fraction):
				raise TypeError(f"{name} must be an int or a Fraction, got {type(value).__name__}")
			object.__setattr__(self, name, Fraction(value))

		if self.onset < 0:
			raise ValueError(f"onset must be non-negative, got {self.onset}")

		if self.duration <= 0:
			raise ValueError(f"duration must be positive, got {self.duration}")

		if not 0 <= self.pitch <= 127:
			raise ValueError(f"pitch must be a MIDI number 0-127, got {self.pitch}")

	@property
	def offset(self) -> Fraction:
		"""End of the note in 8th notes."""
		return self.onset + self.duration


class Section(StrEnum):
	"""Song section of a segment."""

	INTRO = "intro"
	VERSE = "verse"
	PRE_CHORUS = "pre_chorus"
	CHORUS = "chorus"
	BRIDGE = "bridge"
	INSTRUMENTAL = "instrumental"
	OUTRO = "outro"


_SECTION_LABELS = {
	"intro": Section.INTRO,
	"verse": Section.VERSE,
	"pre-chorus": Section.PRE_CHORUS,
	"chorus": Section.CHORUS,
	"chorus lead-out": Section.CHORUS,
	"bridge": Section.BRIDGE,
	"instrumental": Section.INSTRUMENTAL,
	"solo": Section.INSTRUMENTAL,
	"outro": Section.OUTRO,
	"pre-outro": Section.OUTRO,
}


def section_from_label(label: str) -> Section | None:
	"""Section for a Hooktheory label, or None if the label names no known section.

	A trailing number is dropped ("Solo 2") and a compound label takes its first part
	("Intro and Verse").
	"""
	first = label.split(" and ")[0]
	return _SECTION_LABELS.get(re.sub(r"\s+\d+$", "", first).strip().lower())


class Split(StrEnum):
	"""Dataset split, assigned by artist in the source data."""

	TRAIN = "train"
	VALID = "valid"
	TEST = "test"


@dataclass(frozen=True)
class Melody:
	"""Everything a model reads: a monophonic melody, its meter, local keys, section, and tempo."""

	notes: tuple[Note, ...]
	meter: Meter
	keys: tuple[KeySpan, ...]
	section: Section | None
	tempo: int | None
	length: int

	def __post_init__(self) -> None:
		if self.length < 1:
			raise ValueError(f"length must be positive, got {self.length}")

		if self.tempo is not None and self.tempo < 1:
			raise ValueError(f"tempo must be positive, got {self.tempo}")

		starts = [span.start for span in self.keys]
		if not starts or starts[0] != 0:
			raise ValueError("keys must start at position 0")

		if any(a >= b for a, b in pairwise(starts)) or starts[-1] >= self.length:
			raise ValueError(f"key starts must increase within the melody, got {starts}")

		if any(a.offset > b.onset for a, b in pairwise(self.notes)):
			raise ValueError("notes must be sorted and must not overlap")

		if self.notes and self.notes[-1].offset > self.length:
			raise ValueError(f"notes must end within length {self.length}")

	def key_at(self, position: Fraction | int) -> Key:
		"""Key in effect at `position` 8th notes."""
		index = bisect_right([span.start for span in self.keys], position) - 1
		return self.keys[index].key


@dataclass(frozen=True)
class Segment:
	"""A melody with its human accompaniment, the unit of the dataset.

	`events` tile the melody: their durations sum to its length, and no-chord events are merged.
	"""

	id: str
	split: Split
	melody: Melody
	events: tuple[ChordEvent, ...]

	def __post_init__(self) -> None:
		total = sum(event.duration for event in self.events)
		if total != self.melody.length:
			raise ValueError(
				f"event durations sum to {total}, melody length is {self.melody.length}"
			)

		if any(a.chord is None and b.chord is None for a, b in pairwise(self.events)):
			raise ValueError("consecutive no-chord events must be merged")


def merge_repeats(events: Sequence[ChordEvent], melody: Melody) -> tuple[ChordEvent, ...]:
	"""`events` of `melody` with each chord repeated inside a bar held instead of struck again."""
	merged: list[ChordEvent] = []
	start = position = 0

	for event in events:
		last = merged[-1] if merged else None
		if (
			last is not None
			and event.chord is not None
			and event.chord == last.chord
			and position % melody.meter.bar_eighths != 0
			and melody.key_at(position) == melody.key_at(start)
			and last.duration + event.duration <= MAX_DURATION
		):
			merged[-1] = ChordEvent(event.chord, last.duration + event.duration)
		else:
			merged.append(event)
			start = position

		position += event.duration

	return tuple(merged)
