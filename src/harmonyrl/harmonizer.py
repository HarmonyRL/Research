"""Harmonizing a melody, or filling chosen bars between chords that are kept."""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
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
from harmonyrl.schema import (
	MAX_DURATION,
	ChordEvent,
	KeySpan,
	LeadSheet,
	Melody,
	Note,
	RawChord,
	SheetPart,
	chord_pitch_classes,
	merge_repeats,
	project,
)

MAX_BARS = 32
"""Most bars 1 call harmonizes."""

MAX_OVERLAP = 8
"""Most bars of chords already chosen that a window of a long part repeats as context."""

_UNCHOSEN = RawChord(root=1, type=5)
"""Stands in for chords not chosen yet; refilled before it could sound."""


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

	def harmonize_sheet(
		self,
		sheet: LeadSheet,
		*,
		bars: range | None = None,
		candidates: int = 1,
		seed: int | None = None,
		overlap: int = 4,
		**overrides: Any,
	) -> list[LeadSheet]:
		"""Up to `candidates` different harmonizations of `sheet`, whole or only `bars`, which
		count across its parts from 0."""
		if candidates < 1:
			raise ValueError(f"candidates must be at least 1, got {candidates}")
		if not 0 <= overlap <= MAX_OVERLAP:
			raise ValueError(f"overlap must be 0-{MAX_OVERLAP}, got {overlap}")

		counts = [_bar_count(part.melody) for part in sheet.parts]
		if bars is not None and (bars.step != 1 or not 0 <= bars.start < bars.stop <= sum(counts)):
			raise ValueError(
				f"bars must be a nonempty range with step 1 within 0..{sum(counts)}, got {bars}"
			)

		results: list[LeadSheet] = []
		for index in range(candidates):
			draws = random.Random(f"{seed}/{index}") if seed is not None else None
			parts: list[SheetPart] = []
			first = 0

			for part, count in zip(sheet.parts, counts, strict=True):
				low, high = (0, count) if bars is None else (bars.start - first, bars.stop - first)
				low, high = max(low, 0), min(high, count)
				if low >= high:
					parts.append(part)
				else:
					chords = None if bars is None else part.chords
					if chords is None and (low, high) != (0, count):
						raise ValueError(
							f"the part at bars {first}-{first + count - 1} has no chords to keep, "
							"so it can only be filled whole"
						)
					filled = self._fill(part.melody, chords, low, high, overlap, draws, overrides)
					parts.append(SheetPart(part.melody, filled))
				first += count

			results.append(replace(sheet, parts=tuple(parts)))

		return list(dict.fromkeys(results))

	def _fill(
		self,
		melody: Melody,
		chords: tuple[ChordEvent, ...] | None,
		low: int,
		high: int,
		overlap: int,
		draws: random.Random | None,
		overrides: dict[str, Any],
	) -> tuple[ChordEvent, ...]:
		"""Chords of `melody` with bars `low` to `high` filled again, in windows of at most
		`MAX_BARS` bars that start `overlap` bars before the bars they fill."""
		bar = melody.meter.bar_eighths
		count = _bar_count(melody)
		current = chords if chords is not None else _unchosen(melody.length)
		step = MAX_BARS - 2 * overlap
		position = low

		while position < high:
			stop = min(high, position + step)
			start = max(0, position - overlap)
			end = stop
			if stop == high and chords is not None:
				end = min(count, high + overlap, start + MAX_BARS)

			window_start, window_end = start * bar, min(end * bar, melody.length)
			drawn = self.harmonize(
				_slice(melody, window_start, window_end),
				chords=_cut(current, window_start, window_end),
				bars=range(position - start, stop - start),
				seed=None if draws is None else draws.randrange(2**32),
				**overrides,
			)[0]
			current = (
				_cut(current, 0, window_start) + drawn + _cut(current, window_end, melody.length)
			)
			position = stop

		return merge_repeats(current, melody)

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


def _bar_count(melody: Melody) -> int:
	return ceil(melody.length / melody.meter.bar_eighths)


def _unchosen(length: int) -> tuple[ChordEvent, ...]:
	return tuple(
		ChordEvent(_UNCHOSEN, min(MAX_DURATION, length - start))
		for start in range(0, length, MAX_DURATION)
	)


def _slice(melody: Melody, start: int, end: int) -> Melody:
	"""The part of `melody` from `start` to `end`, notes cut at both ends."""
	low, high = Fraction(start), Fraction(end)
	notes = tuple(
		Note(max(note.onset, low) - low, min(note.offset, high) - max(note.onset, low), note.pitch)
		for note in melody.notes
		if note.onset < high and note.offset > low
	)
	keys = (
		KeySpan(0, melody.key_at(start)),
		*(
			KeySpan(span.start - start, span.key)
			for span in melody.keys
			if start < span.start < end
		),
	)

	return replace(melody, notes=notes, keys=keys, length=end - start)


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
