"""Accompaniments as spans and frames, and the chord-tone measure that perturbations keep."""

from collections.abc import Sequence

import numpy as np

from harmonyrl.schema import MAX_DURATION, ChordEvent, Key, Melody, RawChord, Segment, project

from ..features import melody_features
from .pool import ChordPool, tones

type Span = tuple[int, int, RawChord | None]

STRONG = 2
"""Beat strength from which a position counts as strong."""

TOLERANCE = 1e-6


def to_spans(segment: Segment) -> list[Span]:
	"""Start, end, and chord of each event."""
	spans: list[Span] = []
	start = 0

	for event in segment.events:
		spans.append((start, start + event.duration, event.chord))
		start += event.duration

	return spans


def to_events(spans: Sequence[Span]) -> tuple[ChordEvent, ...]:
	"""Events of `spans`, a span longer than the maximum duration split into several."""
	events: list[ChordEvent] = []

	for start, end, chord in spans:
		remaining = end - start
		while remaining > MAX_DURATION:
			events.append(ChordEvent(chord, MAX_DURATION))
			remaining -= MAX_DURATION

		events.append(ChordEvent(chord, remaining))

	return tuple(events)


def to_frames(segment: Segment) -> tuple[list[RawChord | None], list[bool]]:
	"""The chord of each 8th-note frame, and whether an event starts there."""
	chords: list[RawChord | None] = []
	onsets: list[bool] = []

	for event in segment.events:
		chords.extend([event.chord] * event.duration)
		onsets.extend([True] + [False] * (event.duration - 1))

	return chords, onsets


def from_frames(chords: Sequence[RawChord | None], onsets: Sequence[bool]) -> list[Span]:
	"""Spans that start at each kept onset and wherever the chord changes; no chord merges."""
	spans: list[Span] = []

	for t, (chord, onset) in enumerate(zip(chords, onsets, strict=True)):
		if spans and chord == spans[-1][2] and (chord is None or not onset):
			spans[-1] = (spans[-1][0], t + 1, chord)
		else:
			spans.append((t, t + 1, chord))

	return spans


def projected(segment: Segment, events: Sequence[ChordEvent]) -> list[tuple[RawChord | None, int]]:
	"""Each event as its projected chord and duration, to compare accompaniments."""
	result = []
	start = 0

	for event in events:
		chord = event.chord and project(event.chord, segment.melody.key_at(start))
		result.append((chord, event.duration))
		start += event.duration

	return result


def index_at(spans: Sequence[Span], position: int) -> int:
	"""Index of the span sounding at `position`."""
	return next(i for i, (start, end, _) in enumerate(spans) if start <= position < end)


def different(melody: Melody, a: Span, b: Span) -> bool:
	"""Whether 2 spans both carry chords and the chords differ once projected."""
	if a[2] is None or b[2] is None:
		return False

	return project(a[2], melody.key_at(a[0])) != project(b[2], melody.key_at(b[0]))


def profiles(melody: Melody) -> np.ndarray:
	"""Time each pitch class sounds in each 8th-note frame (frames, 12)."""
	relative = melody_features(melody).pitch_classes.astype(np.float64)
	return np.stack([np.roll(relative[t], melody.key_at(t).tonic) for t in range(melody.length)])


def fit(frame_profiles: np.ndarray, start: int, end: int, chord_tones: np.ndarray) -> float:
	"""Melody time on `chord_tones` from `start` to `end`."""
	return float(frame_profiles[start:end].sum(axis=0) @ chord_tones)


def frame_fit(
	frame_profiles: np.ndarray,
	frames: Sequence[RawChord | None],
	key: Key,
	start: int,
	end: int,
	source: int,
) -> float:
	"""Melody time on chord tones from `start` to `end` under the chords from `source` onward."""
	return sum(
		float(frame_profiles[start + k] @ tones(frames[source + k], key))
		for k in range(end - start)
	)


def replace(
	melody: Melody,
	frame_profiles: np.ndarray,
	spans: list[Span],
	i: int,
	pool: ChordPool,
	rng: np.random.Generator,
) -> bool:
	"""Give span `i` another pool chord that keeps its chord tones, sampled by frequency."""
	start, end, original = spans[i]
	key = melody.key_at(start)

	if original is None or key.mode not in pool.chords:
		return False

	profile = frame_profiles[start:end].sum(axis=0)
	fits = np.roll(pool.tones[key.mode], key.tonic, axis=1) @ profile
	baseline = float(tones(original, key) @ profile)
	label = project(original, key)
	eligible = [
		c
		for c, chord in enumerate(pool.chords[key.mode])
		if chord != label and fits[c] >= baseline - TOLERANCE
	]

	if not eligible:
		return False

	weights = pool.weights[key.mode][eligible]
	choice = int(rng.choice(eligible, p=weights / weights.sum()))
	spans[i] = (start, end, pool.chords[key.mode][choice])

	return True
