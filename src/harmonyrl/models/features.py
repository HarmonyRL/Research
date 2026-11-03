"""Model inputs and training targets derived from schema types. A frame is one 8th note."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, log2

import numpy as np

from harmonyrl.schema import (
	HEAD_VALUES,
	ChordEvent,
	Melody,
	Mode,
	Section,
	Segment,
	allowed,
	no_chord_limit,
	project,
)

BAR_POSITIONS = 24
"""Distinct bar positions in 8th notes. Later positions in longer bars share the last value."""

MAX_BEATS_PER_BAR = 16
"""Larger beat counts share the value of this one."""

OCTAVES = 11
"""MIDI octaves `pitch // 12`."""

BEAT_STRENGTHS = 4
"""Values of `Meter.beat_strength`."""

TEMPO_BUCKETS = 16
"""Tempo buckets a quarter octave wide in 8th notes per minute, from 64 up.
Faster tempos share the last bucket."""

MODES = tuple(Mode)
SECTIONS = tuple(Section)

IGNORE = -100
"""Target index of a head that is not predicted, the default `ignore_index` of torch losses."""

_INDEX: dict[str, dict[object, int]] = {
	head: {value: i for i, value in enumerate(values)} for head, values in HEAD_VALUES.items()
}


@dataclass(frozen=True)
class MelodyFeatures:
	"""Per-frame inputs for one melody, and its meter, section, and tempo."""

	pitch_classes: np.ndarray
	onset: np.ndarray
	octave: np.ndarray
	bar_position: np.ndarray
	beat_strength: np.ndarray
	mode: np.ndarray
	key_start: np.ndarray
	beats_per_bar: int
	beat_unit: int
	section: int
	tempo: int


def melody_features(melody: Melody) -> MelodyFeatures:
	"""Frame features of `melody`, which may come from the dataset or from an imported file."""
	length = melody.length
	meter = melody.meter

	tonic = np.empty(length, dtype=np.int64)
	mode = np.empty(length, dtype=np.int64)
	key_start = np.zeros(length, dtype=bool)
	ends = [span.start for span in melody.keys[1:]] + [length]

	for span, end in zip(melody.keys, ends, strict=True):
		tonic[span.start : end] = span.key.tonic
		mode[span.start : end] = MODES.index(span.key.mode)
		key_start[span.start] = True

	pitch_classes = np.zeros((length, 12), dtype=np.float32)
	onset = np.zeros(length, dtype=bool)
	octave = np.zeros(length, dtype=np.int64)
	longest = np.zeros(length, dtype=np.float32)

	for note in melody.notes:
		onset[floor(note.onset)] = True
		for frame in range(floor(note.onset), ceil(note.offset)):
			overlap = float(min(note.offset, frame + 1) - max(note.onset, frame))
			pitch_classes[frame, (note.pitch - tonic[frame]) % 12] += overlap
			if overlap > longest[frame]:
				longest[frame] = overlap
				octave[frame] = note.pitch // 12 + 1

	positions = np.arange(length)
	return MelodyFeatures(
		pitch_classes=pitch_classes,
		onset=onset,
		octave=octave,
		bar_position=np.minimum(positions % meter.bar_eighths, BAR_POSITIONS - 1),
		beat_strength=np.array([meter.beat_strength(p) for p in range(length)], dtype=np.int64),
		mode=mode,
		key_start=key_start,
		beats_per_bar=min(meter.beats_per_bar, MAX_BEATS_PER_BAR),
		beat_unit=0 if meter.beat_unit == 4 else 1,
		section=0 if melody.section is None else SECTIONS.index(melody.section) + 1,
		tempo=0 if melody.tempo is None else _tempo_bucket(melody.tempo, meter.beat_unit) + 1,
	)


@dataclass(frozen=True)
class EventTargets:
	"""Training targets for the chord events of one segment.

	`values[i, h]` indexes `HEAD_VALUES` for head h of event i, or is `IGNORE` for the chord heads
	of a no-chord event. `masks[head][i]` is the action mask that head was predicted under.
	"""

	starts: np.ndarray
	values: np.ndarray
	masks: dict[str, np.ndarray]


def event_targets(segment: Segment, *, restrict_no_chord: bool = False) -> EventTargets:
	"""Targets of the projected chords of `segment`.

	With `restrict_no_chord`, no chord is allowed only as the 1st event, within the
	`no_chord_limit` of the melody. Raises ValueError if a target falls outside its mask,
	which means the segment breaks the contract.
	"""
	try:
		return prefix_targets(segment.melody, segment.events, restrict_no_chord=restrict_no_chord)
	except ValueError as error:
		raise ValueError(f"{segment.id}: {error}") from error


def prefix_targets(
	melody: Melody, events: Sequence[ChordEvent], *, restrict_no_chord: bool = False
) -> EventTargets:
	"""Targets of `events`, the opening events of a harmonization of `melody`, as `event_targets`
	builds them."""
	heads = tuple(HEAD_VALUES)
	count = len(events)
	starts = np.zeros(count, dtype=np.int64)
	values = np.full((count, len(heads)), IGNORE, dtype=np.int64)
	masks = {head: np.ones((count, len(v)), dtype=bool) for head, v in HEAD_VALUES.items()}

	limit = no_chord_limit(melody)
	start = 0
	previous_no_chord = False

	for i, event in enumerate(events):
		starts[i] = start
		no_chord_room = (0 if i else limit) if restrict_no_chord else None
		target: dict[str, object] = {"root": None}

		if event.chord is not None:
			chord = project(event.chord, melody.key_at(start))
			target = {head: getattr(chord, head) for head in heads if head != "duration"}

		target["duration"] = event.duration

		for h, head in enumerate(heads):
			if head not in target:
				continue

			mask = allowed(
				head,
				target,
				previous_no_chord=previous_no_chord,
				remaining=melody.length - start,
				no_chord_room=no_chord_room,
			)
			index = _INDEX[head][target[head]]
			if not mask[index]:
				raise ValueError(f"event {i} has masked {head} {target[head]!r}")

			masks[head][i] = mask
			values[i, h] = index

		previous_no_chord = event.chord is None
		start += event.duration

	return EventTargets(starts=starts, values=values, masks=masks)


def _tempo_bucket(tempo: int, beat_unit: int) -> int:
	eighths_per_minute = tempo * (3 if beat_unit == 8 else 2)
	return min(max(round(4 * log2(eighths_per_minute / 64)), 0), TEMPO_BUCKETS - 1)
