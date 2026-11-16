"""Harmonizing a melody, or filling chosen bars between chords that are kept."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from math import ceil, log
from pathlib import Path
from typing import Any, Self

import torch

from harmonyrl.models import (
	ChordPolicy,
	GenerationConfig,
	collate,
	melody_features,
	prefix_targets,
)
from harmonyrl.schema import ChordEvent, Melody, chord_pitch_classes, merge_repeats, project

MAX_BARS = 32
"""Most bars 1 call harmonizes."""


class HarmonyRL:
	"""A chord policy with the settings it generates with by default."""

	def __init__(self, policy: ChordPolicy, generation_config: GenerationConfig) -> None:
		self.policy = policy.eval()
		self.generation_config = generation_config

	@classmethod
	def from_pretrained(
		cls, path_or_repo: str | Path, *, device: str | None = None, revision: str | None = None
	) -> Self:
		"""Load a released model from a local directory or a Hub repository."""
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		policy = ChordPolicy.from_pretrained(path_or_repo, revision=revision)
		config = GenerationConfig.from_pretrained(path_or_repo, revision=revision)

		return cls(policy.to(device), config)

	def harmonize(
		self,
		melody: Melody,
		*,
		chords: Sequence[ChordEvent] | None = None,
		bars: range | None = None,
		candidates: int = 1,
		seed: int | None = None,
		**overrides: Any,
	) -> list[tuple[ChordEvent, ...]]:
		config = replace(self.generation_config, **overrides)
		_check_length(melody)
		if candidates < 1:
			raise ValueError(f"candidates must be at least 1, got {candidates}")

		device = next(self.policy.parameters()).device
		generator = None if seed is None else torch.Generator(device).manual_seed(seed)

		if chords is None:
			drafts = self.policy.generate(
				[melody] * candidates, temperature=config.temperature, generator=generator
			)

			return list(dict.fromkeys(merge_repeats(draft, melody) for draft in drafts))

		chords = tuple(chords)
		self._check_chords(melody, chords)
		start, stop = _bar_span(melody, bars)
		before, after = _cut(chords, 0, start), _cut(chords, stop, melody.length)

		pool = candidates * config.pool
		drafts = self.policy.generate(
			[melody] * pool,
			prefixes=[before] * pool,
			temperature=config.temperature,
			generator=generator,
		)
		counts = Counter(_cut(draft, start, stop) for draft in drafts)
		middles = list(counts)
		weights = torch.tensor([log(counts[middle]) for middle in middles], device=device)

		# The policy reads left to right, so the chords after the bars weigh each draft instead.
		if after:
			weights += self._log_probs(melody, [before + middle + after for middle in middles])
			weights -= self._log_probs(melody, [before + middle for middle in middles])

		picks = torch.multinomial(
			weights.softmax(0), min(candidates, len(middles)), generator=generator
		)
		results = list(
			dict.fromkeys(
				merge_repeats(before + middles[i] + after, melody) for i in picks.tolist()
			)
		)

		kept = _sounding(melody, chords)
		for events in results:
			sounding = _sounding(melody, events)
			if sounding[:start] != kept[:start] or sounding[stop:] != kept[stop:]:
				raise RuntimeError("a bar outside the filled bars changed its chords")

		return results

	def _check_chords(self, melody: Melody, chords: tuple[ChordEvent, ...]) -> None:
		if sum(event.duration for event in chords) != melody.length:
			raise ValueError(f"chords must cover the melody's {melody.length} 8th notes exactly")

		try:
			prefix_targets(melody, chords, restrict_no_chord=self.policy.config.restrict_no_chord)
		except ValueError as error:
			raise ValueError(f"chords the policy cannot follow: {error}") from error

	@torch.no_grad()
	def _log_probs(
		self, melody: Melody, harmoniozations: Sequence[tuple[ChordEvent, ...]]
	) -> torch.Tensor:
		features = melody_features(melody)
		restrict = self.policy.config.restrict_no_chord
		batch = collate(
			[
				(features, prefix_targets(melody, h, restrict_no_chord=restrict))
				for h in harmoniozations
			]
		)
		device = next(self.policy.parameters()).device
		return self.policy(batch.to(device)).log_prob.sum(dim=1)


def _check_length(melody: Melody) -> None:
	bars = ceil(melody.length / melody.meter.bar_eighths)
	if bars > MAX_BARS:
		raise ValueError(f"the melody has {bars} bars, more than {MAX_BARS}")


def _bar_span(melody: Melody, bars: range | None) -> tuple[int, int]:
	if bars is None:
		return 0, melody.length

	bar = melody.meter.bar_eighths
	count = ceil(melody.length / bar)
	if bars.step != 1 or not 0 <= bars.start < bars.stop <= count:
		raise ValueError(f"bars must be a nonempty range with step 1 within 0..{count}, got {bars}")

	return bars.start * bar, min(bars.stop * bar, melody.length)


def _cut(events: Sequence[ChordEvent], start: int, end: int) -> tuple[ChordEvent, ...]:
	cut: list[ChordEvent] = []
	position = 0

	for event in events:
		low, high = max(position, start), min(position + event.duration, end)
		if low < high:
			cut.append(ChordEvent(event.chord, high - low))

		position += event.duration

	return tuple(cut)


def _sounding(
	melody: Melody, events: Sequence[ChordEvent]
) -> list[tuple[int, frozenset[int]] | None]:
	sounding: list[tuple[int, frozenset[int]] | None] = []
	position = 0

	for event in events:
		value = None
		if event.chord is not None:
			key = melody.key_at(position)
			value = chord_pitch_classes(project(event.chord, key), key)

		sounding += [value] * event.duration
		position += event.duration

	return sounding
