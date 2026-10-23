"""Hidden Markov model baseline that harmonizes a melody with 1 chord per slot.

Follows the HMM of Yeh et al. (2021), "Automatic melody harmonization with triad chords: A
comparative study": chord labels are hidden states, the pitch-class profile of the melody in each
slot is a diagonal Gaussian emission, and transitions interpolate bigram and prior probabilities.
Here labels are model-vocabulary chords relative to the local key, so each mode has its own
parameters, and a slot spans the time from 1 strong beat to the next, a half bar in 4/4.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from harmonyrl.schema import (
	HEAD_VALUES,
	MAX_DURATION,
	ChordEvent,
	Key,
	Melody,
	Mode,
	RawChord,
	Segment,
	project,
)

from .features import melody_features

_CHORD_HEADS = tuple(head for head in HEAD_VALUES if head != "duration")

type Label = RawChord | None


@dataclass(frozen=True, eq=False)
class ModeParameters:
	"""HMM parameters for the keys of 1 mode.

	`prior` (labels,) and each row of `bigram` (labels, labels) sum to 1. `mean` and `variance`
	(labels, 12) describe the pitch-class profiles of the slots each label was given.
	"""

	labels: tuple[Label, ...]
	prior: np.ndarray
	bigram: np.ndarray
	mean: np.ndarray
	variance: np.ndarray


@dataclass(frozen=True, eq=False)
class ChordHMM:
	"""Chord labels per slot, decoded with Viterbi.

	The transition is `(1 - beta) * prior + beta * bigram`. Where the key changes between slots,
	the labels refer to different keys, so the new slot takes the prior of its mode instead.
	`min_variance` floors the emission variances.
	"""

	beta: float
	min_variance: float
	modes: dict[Mode, ModeParameters]

	def __post_init__(self) -> None:
		if not 0 <= self.beta <= 1:
			raise ValueError(f"beta must be in [0, 1], got {self.beta}")

		if self.min_variance <= 0:
			raise ValueError(f"min_variance must be positive, got {self.min_variance}")

	@classmethod
	def fit(
		cls, segments: Iterable[Segment], *, min_count: int, beta: float, min_variance: float
	) -> ChordHMM:
		"""Estimate the parameters from human accompaniments.

		Each slot is labeled with the projected chord that sounds longest in it.
		Labels given to fewer than `min_count` slots of a mode are not states of that mode,
		and their slots are left out of the estimates.
		"""
		counts: dict[Mode, Counter[Label]] = defaultdict(Counter)
		bigrams: dict[Mode, Counter[tuple[Label, Label]]] = defaultdict(Counter)
		profiles: dict[Mode, dict[Label, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

		for segment in segments:
			bounds = _slots(segment.melody)
			keys = [segment.melody.key_at(start) for start, _ in bounds]
			labels = _human_labels(segment, bounds)

			for s, (key, label, profile) in enumerate(
				zip(keys, labels, _profiles(segment.melody, bounds), strict=True)
			):
				counts[key.mode][label] += 1
				profiles[key.mode][label].append(profile)
				if s and keys[s - 1] == key:
					bigrams[key.mode][labels[s - 1], label] += 1

		modes: dict[Mode, ModeParameters] = {}
		for mode, mode_counts in counts.items():
			states = tuple(
				label for label, count in mode_counts.most_common() if count >= min_count
			)
			if not states:
				continue

			index = {label: i for i, label in enumerate(states)}
			prior = np.array([mode_counts[label] for label in states], dtype=np.float64)
			bigram = np.zeros((len(states), len(states)))

			for (previous, label), count in bigrams[mode].items():
				if previous in index and label in index:
					bigram[index[previous], index[label]] = count

			totals = bigram.sum(axis=1, keepdims=True)
			bigram = np.divide(
				bigram, totals, out=np.full_like(bigram, 1 / len(states)), where=totals > 0
			)
			stacked = [np.stack(profiles[mode][label]) for label in states]
			modes[mode] = ModeParameters(
				labels=states,
				prior=prior / prior.sum(),
				bigram=bigram,
				mean=np.stack([p.mean(axis=0) for p in stacked]),
				variance=np.stack([p.var(axis=0) for p in stacked]),
			)

		return cls(beta=beta, min_variance=min_variance, modes=modes)

	def harmonize(self, melody: Melody) -> tuple[ChordEvent, ...]:
		"""The most likely chord per slot, as events that tile `melody`.

		Equal labels in a row merge into 1 event, split where an event would exceed the maximum
		duration. No events express a longer stretch without a chord, so such a stretch is decoded
		again with no chord ruled out on its slots.
		Raises ValueError for a mode without parameters.
		"""
		bounds = _slots(melody)
		keys = [melody.key_at(start) for start, _ in bounds]
		profiles = _profiles(melody, bounds)
		no_chord_ruled_out: set[int] = set()

		while True:
			runs = _runs(self._viterbi(profiles, keys, no_chord_ruled_out), bounds, keys)
			too_long = {
				s
				for label, duration, slots in runs
				if label is None and duration > MAX_DURATION
				for s in slots
			}
			if not too_long:
				return _events(runs)

			no_chord_ruled_out |= too_long

	def save(self, path: Path) -> None:
		"""Write the parameters and the vocabulary they index to an .npz file."""
		arrays: dict[str, np.ndarray] = {
			"head_values": np.array(json.dumps(HEAD_VALUES)),
			"settings": np.array(
				json.dumps({"beta": self.beta, "min_variance": self.min_variance})
			),
		}

		for mode, parameters in self.modes.items():
			arrays[f"{mode.value}/labels"] = np.array(
				[_encode(label) for label in parameters.labels]
			)
			arrays[f"{mode.value}/prior"] = parameters.prior
			arrays[f"{mode.value}/bigram"] = parameters.bigram
			arrays[f"{mode.value}/mean"] = parameters.mean
			arrays[f"{mode.value}/variance"] = parameters.variance

		np.savez(path, allow_pickle=False, **arrays)

	@classmethod
	def load(cls, path: Path) -> ChordHMM:
		"""Read parameters written by `save`. Raises ValueError if the vocabulary has changed."""
		with np.load(path) as data:
			if str(data["head_values"]) != json.dumps(HEAD_VALUES):
				raise ValueError(f"{path} was saved with a different HEAD_VALUES")

			modes = {
				mode: ModeParameters(
					labels=tuple(_decode(row) for row in data[f"{mode.value}/labels"]),
					prior=data[f"{mode.value}/prior"],
					bigram=data[f"{mode.value}/bigram"],
					mean=data[f"{mode.value}/mean"],
					variance=data[f"{mode.value}/variance"],
				)
				for mode in Mode
				if f"{mode.value}/labels" in data.files
			}

			return cls(**json.loads(str(data["settings"])), modes=modes)

	@cached_property
	def _log_parameters(self) -> dict[Mode, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
		"""Log prior, log transition, emission mean, and floored variance per mode."""
		return {
			mode: (
				np.log(p.prior),
				np.log((1 - self.beta) * p.prior + self.beta * p.bigram),
				p.mean,
				np.maximum(p.variance, self.min_variance),
			)
			for mode, p in self.modes.items()
		}

	def _viterbi(
		self, profiles: np.ndarray, keys: Sequence[Key], no_chord_ruled_out: set[int]
	) -> list[Label]:
		scores: list[np.ndarray] = []
		pointers: list[np.ndarray] = []

		for s, (profile, key) in enumerate(zip(profiles, keys, strict=True)):
			if key.mode not in self.modes:
				raise ValueError(f"no parameters for mode {key.mode}")

			log_prior, log_transition, mean, variance = self._log_parameters[key.mode]
			emission = -0.5 * (np.log(2 * np.pi * variance) + (profile - mean) ** 2 / variance).sum(
				axis=1
			)
			labels = self.modes[key.mode].labels

			if s in no_chord_ruled_out and None in labels:
				emission[labels.index(None)] = -np.inf

			if s and keys[s - 1] == key:
				candidates = scores[-1][:, None] + log_transition
				pointers.append(candidates.argmax(axis=0))
				scores.append(candidates.max(axis=0) + emission)
			else:
				best = int(scores[-1].argmax()) if s else 0
				pointers.append(np.full(len(log_prior), best))
				scores.append((scores[-1][best] if s else 0.0) + log_prior + emission)

		path = [int(scores[-1].argmax())]
		for s in range(len(scores) - 1, 0, -1):
			path.append(int(pointers[s][path[-1]]))
		path.reverse()

		return [self.modes[key.mode].labels[i] for key, i in zip(keys, path, strict=True)]


def _slots(melody: Melody) -> list[tuple[int, int]]:
	meter = melody.meter
	offsets = [beat * meter.eighths_per_beat for beat in meter.strong_beats()]
	starts = [
		bar + offset
		for bar in range(0, melody.length, meter.bar_eighths)
		for offset in offsets
		if bar + offset < melody.length
	]

	return list(zip(starts, [*starts[1:], melody.length], strict=True))


def _profiles(melody: Melody, bounds: Sequence[tuple[int, int]]) -> np.ndarray:
	pitch_classes = melody_features(melody).pitch_classes
	return np.stack([pitch_classes[start:end].sum(axis=0) / (end - start) for start, end in bounds])


def _human_labels(segment: Segment, bounds: Sequence[tuple[int, int]]) -> list[Label]:
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


def _runs(
	labels: Sequence[Label], bounds: Sequence[tuple[int, int]], keys: Sequence[Key]
) -> list[tuple[Label, int, list[int]]]:
	"""Label, total duration, and slots of each stretch that becomes 1 event before splitting.

	A chord continues only within 1 key, since it is read relative to the key where it starts.
	"""
	runs: list[tuple[Label, int, list[int]]] = []
	for s, (label, (start, end)) in enumerate(zip(labels, bounds, strict=True)):
		if runs and runs[-1][0] == label and (label is None or keys[s - 1] == keys[s]):
			runs[-1] = (label, runs[-1][1] + end - start, [*runs[-1][2], s])
		else:
			runs.append((label, end - start, [s]))

	return runs


def _events(runs: Sequence[tuple[Label, int, list[int]]]) -> tuple[ChordEvent, ...]:
	events: list[ChordEvent] = []

	for label, duration, _ in runs:
		remaining = duration

		while remaining > MAX_DURATION:
			events.append(ChordEvent(label, MAX_DURATION))
			remaining -= MAX_DURATION

		events.append(ChordEvent(label, remaining))

	return tuple(events)


def _encode(label: Label) -> list[int]:
	if label is None:
		return [0] * len(_CHORD_HEADS)

	return [HEAD_VALUES[head].index(getattr(label, head)) for head in _CHORD_HEADS]


def _decode(row: np.ndarray) -> Label:
	values: dict[str, Any] = {
		head: HEAD_VALUES[head][int(i)] for head, i in zip(_CHORD_HEADS, row, strict=True)
	}

	if values["root"] is None:
		return None

	return RawChord(**values)
