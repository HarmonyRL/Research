"""Slots, pitch-class profiles, Viterbi decoding, and chord events for the HMM baseline."""

from collections.abc import Sequence

import numpy as np

from harmonyrl.schema import MAX_DURATION, ChordEvent, Key, Melody, RawChord, Segment, project

from ..features import melody_features

type Label = RawChord | None
type Run = tuple[Label, int, list[int]]


def slots(melody: Melody) -> list[tuple[int, int]]:
	"""Start and end of each slot, from 1 strong beat to the next. The last ends with `melody`."""
	meter = melody.meter
	offsets = [beat * meter.eighths_per_beat for beat in meter.strong_beats()]
	starts = [
		bar + offset
		for bar in range(0, melody.length, meter.bar_eighths)
		for offset in offsets
		if bar + offset < melody.length
	]

	return list(zip(starts, [*starts[1:], melody.length], strict=True))


def profiles(melody: Melody, bounds: Sequence[tuple[int, int]]) -> np.ndarray:
	"""Share of each slot (slots, 12) that each pitch class sounds, relative to the local tonic."""
	pitch_classes = melody_features(melody).pitch_classes
	return np.stack([pitch_classes[start:end].sum(axis=0) / (end - start) for start, end in bounds])


def human_labels(segment: Segment, bounds: Sequence[tuple[int, int]]) -> list[Label]:
	"""The projected chord that sounds longest in each slot, the earliest on a tie."""
	spans: list[tuple[int, int, Label]] = []
	start = 0

	for event in segment.events:
		chord = None if event.chord is None else project(event.chord, segment.melody.key_at(start))
		spans.append((start, start + event.duration, chord))
		start += event.duration

	return [
		max(spans, key=lambda span: min(span[1], end) - max(span[0], begin))[2]
		for begin, end in bounds
	]


def viterbi(
	emissions: Sequence[np.ndarray],
	log_priors: Sequence[np.ndarray],
	log_transitions: Sequence[np.ndarray | None],
) -> list[int]:
	"""Index of the label of each slot on the most likely path.

	Slot s has its own labels, scored by `emissions[s]`. `log_transitions[s]`
	(labels of s - 1, labels of s) links it to the slot before.
	Where it is None, slot s starts afresh from `log_priors[s]` after the best path so far,
	as the 1st slot must.
	"""
	if log_transitions and log_transitions[0] is not None:
		raise ValueError("the 1st slot has no slot before it to transition from")

	scores: list[np.ndarray] = []
	pointers: list[np.ndarray] = []

	for emission, log_prior, log_transition in zip(
		emissions, log_priors, log_transitions, strict=True
	):
		if log_transition is not None:
			candidates = scores[-1][:, None] + log_transition
			pointers.append(candidates.argmax(axis=0))
			scores.append(candidates.max(axis=0) + emission)
		else:
			best = int(scores[-1].argmax()) if scores else 0
			pointers.append(np.full(len(emission), best))
			scores.append((scores[-1][best] if scores else 0.0) + log_prior + emission)

	path = [int(scores[-1].argmax())]
	for s in range(len(scores) - 1, 0, -1):
		path.append(int(pointers[s][path[-1]]))
	path.reverse()

	return path


def runs(
	labels: Sequence[Label], bounds: Sequence[tuple[int, int]], keys: Sequence[Key]
) -> list[Run]:
	"""Label, total duration, and slots of each stretch that becomes 1 event before splitting.

	A chord continues only within 1 key, since it is read relative to the key where it starts.
	"""
	merged: list[Run] = []

	for s, (label, (start, end)) in enumerate(zip(labels, bounds, strict=True)):
		if merged and merged[-1][0] == label and (label is None or keys[s - 1] == keys[s]):
			merged[-1] = (label, merged[-1][1] + end - start, [*merged[-1][2], s])
		else:
			merged.append((label, end - start, [s]))

	return merged


def events(stretches: Sequence[Run]) -> tuple[ChordEvent, ...]:
	"""1 event per stretch, split where it would exceed the maximum duration."""
	result: list[ChordEvent] = []

	for label, duration, _ in stretches:
		remaining = duration

		while remaining > MAX_DURATION:
			result.append(ChordEvent(label, MAX_DURATION))
			remaining -= MAX_DURATION

		result.append(ChordEvent(label, remaining))

	return tuple(result)
