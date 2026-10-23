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
from typing import Any, Self

import numpy as np
from huggingface_hub import ModelHubMixin, hf_hub_download

from harmonyrl.schema import (
	HEAD_VALUES,
	MAX_DURATION,
	ChordEvent,
	Key,
	Melody,
	Mode,
	RawChord,
	Segment,
)

from . import decode
from .decode import Label

PARAMETERS_FILE = "parameters.npz"
_CHORD_HEADS = tuple(head for head in HEAD_VALUES if head != "duration")


@dataclass(frozen=True)
class HMMConfig:
	"""Settings of `ChordHMM`. The defaults follow Yeh et al. (2021) where they give a value.

	A label needs `min_count` slots of a mode to be a state of that mode. The transition is
	`(1 - beta) * prior + beta * bigram`. `min_variance` floors the emission variances, since a
	label seen only on identical slots, such as rests, would otherwise have a variance of 0.
	"""

	min_count: int = 20
	beta: float = 0.08
	min_variance: float = 0.03

	def __post_init__(self) -> None:
		if self.min_count < 1:
			raise ValueError(f"min_count must be positive, got {self.min_count}")

		if not 0 <= self.beta <= 1:
			raise ValueError(f"beta must be in [0, 1], got {self.beta}")

		if self.min_variance <= 0:
			raise ValueError(f"min_variance must be positive, got {self.min_variance}")


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


class ChordHMM(ModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""Chord labels per slot, decoded with Viterbi.

	Where the key changes between slots, the labels refer to different keys, so the new slot takes
	the prior of its mode instead of a transition.
	"""

	def __init__(self, config: HMMConfig, modes: dict[Mode, ModeParameters]) -> None:
		self.config = config
		self.modes = modes

	@classmethod
	def fit(cls, segments: Iterable[Segment], config: HMMConfig) -> Self:
		"""Estimate the parameters from human accompaniments.

		Each slot is labeled with the projected chord that sounds longest in it.
		Labels below `config.min_count` slots of a mode are not states of that mode,
		and their slots are left out of the estimates.
		"""
		counts: dict[Mode, Counter[Label]] = defaultdict(Counter)
		bigrams: dict[Mode, Counter[tuple[Label, Label]]] = defaultdict(Counter)
		profiles: dict[Mode, dict[Label, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

		for segment in segments:
			bounds = decode.slots(segment.melody)
			keys = [segment.melody.key_at(start) for start, _ in bounds]
			labels = decode.human_labels(segment, bounds)

			for s, (key, label, profile) in enumerate(
				zip(keys, labels, decode.profiles(segment.melody, bounds), strict=True)
			):
				counts[key.mode][label] += 1
				profiles[key.mode][label].append(profile)
				if s and keys[s - 1] == key:
					bigrams[key.mode][labels[s - 1], label] += 1

		modes: dict[Mode, ModeParameters] = {}
		for mode, mode_counts in counts.items():
			states = tuple(
				label for label, count in mode_counts.most_common() if count >= config.min_count
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

		return cls(config, modes)

	def harmonize(self, melody: Melody) -> tuple[ChordEvent, ...]:
		"""The most likely chord per slot, as events that tile `melody`.

		Equal labels in a row merge into 1 event, split where an event would exceed
		the maximum duration. No events express a longer stretch without a chord,
		so such a stretch is decoded again with no chord ruled out on its slots.
		Raises `ValueError` for a mode without parameters.
		"""
		bounds = decode.slots(melody)
		keys = [melody.key_at(start) for start, _ in bounds]
		profiles = decode.profiles(melody, bounds)
		no_chord_ruled_out: set[int] = set()

		while True:
			stretches = decode.runs(
				self._best_labels(profiles, keys, no_chord_ruled_out), bounds, keys
			)
			too_long = {
				s
				for label, duration, slots in stretches
				if label is None and duration > MAX_DURATION
				for s in slots
			}
			if not too_long:
				return decode.events(stretches)

			no_chord_ruled_out |= too_long

	def _save_pretrained(self, save_directory: Path) -> None:
		arrays: dict[str, np.ndarray] = {"head_values": np.array(json.dumps(HEAD_VALUES))}

		for mode, parameters in self.modes.items():
			arrays[f"{mode.value}/labels"] = np.array(
				[_label_to_row(label) for label in parameters.labels]
			)
			arrays[f"{mode.value}/prior"] = parameters.prior
			arrays[f"{mode.value}/bigram"] = parameters.bigram
			arrays[f"{mode.value}/mean"] = parameters.mean
			arrays[f"{mode.value}/variance"] = parameters.variance

		np.savez(save_directory / PARAMETERS_FILE, allow_pickle=False, **arrays)

	@classmethod
	def _from_pretrained(
		cls,
		*,
		model_id: str,
		revision: str | None,
		cache_dir: str | Path | None,
		force_download: bool,
		local_files_only: bool,
		token: str | bool | None,
		**model_kwargs: Any,
	) -> Self:
		if Path(model_id).is_dir():
			path = Path(model_id) / PARAMETERS_FILE
		else:
			path = Path(
				hf_hub_download(
					repo_id=model_id,
					filename=PARAMETERS_FILE,
					revision=revision,
					cache_dir=cache_dir,
					force_download=force_download,
					token=token,
					local_files_only=local_files_only,
				)
			)

		with np.load(path) as data:
			if str(data["head_values"]) != json.dumps(HEAD_VALUES):
				raise ValueError(f"{model_id} was saved with a different HEAD_VALUES")

			modes = {
				mode: ModeParameters(
					labels=tuple(_row_to_label(row) for row in data[f"{mode.value}/labels"]),
					prior=data[f"{mode.value}/prior"],
					bigram=data[f"{mode.value}/bigram"],
					mean=data[f"{mode.value}/mean"],
					variance=data[f"{mode.value}/variance"],
				)
				for mode in Mode
				if f"{mode.value}/labels" in data.files
			}

		return cls(model_kwargs["config"], modes)

	@cached_property
	def _log_parameters(self) -> dict[Mode, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
		"""Log prior, log transition, emission mean, and floored variance per mode."""
		beta = self.config.beta
		return {
			mode: (
				np.log(p.prior),
				np.log((1 - beta) * p.prior + beta * p.bigram),
				p.mean,
				np.maximum(p.variance, self.config.min_variance),
			)
			for mode, p in self.modes.items()
		}

	def _best_labels(
		self, profiles: np.ndarray, keys: Sequence[Key], no_chord_ruled_out: set[int]
	) -> list[Label]:
		emissions: list[np.ndarray] = []
		log_priors: list[np.ndarray] = []
		log_transitions: list[np.ndarray | None] = []

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

			emissions.append(emission)
			log_priors.append(log_prior)
			log_transitions.append(log_transition if s and keys[s - 1] == key else None)

		path = decode.viterbi(emissions, log_priors, log_transitions)

		return [self.modes[key.mode].labels[i] for key, i in zip(keys, path, strict=True)]


def _label_to_row(label: Label) -> list[int]:
	if label is None:
		return [0] * len(_CHORD_HEADS)
	return [HEAD_VALUES[head].index(getattr(label, head)) for head in _CHORD_HEADS]


def _row_to_label(row: np.ndarray) -> Label:
	values: dict[str, Any] = {
		head: HEAD_VALUES[head][int(i)] for head, i in zip(_CHORD_HEADS, row, strict=True)
	}

	if values["root"] is None:
		return None
	return RawChord(**values)
