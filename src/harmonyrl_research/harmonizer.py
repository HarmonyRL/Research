"""Harmonizing a melody, or filling chosen bars between chords that are kept."""

import random
from collections.abc import Callable, Hashable, Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import accumulate
from math import ceil
from pathlib import Path
from typing import Any, Self

import torch

from harmonyrl_research.models import (
	ChordPolicy,
	GenerationConfig,
	collate,
	melody_features,
	prefix_targets,
)
from harmonyrl_research.schema import (
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

MAX_CANDIDATES = 16
"""Most candidates 1 call returns."""

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
		"""Up to `candidates` harmonizations of `melody`, whole or only `bars` within `chords`."""
		config = replace(self.generation_config, **overrides)
		HarmonyRL._check_length(melody)
		HarmonyRL._check_candidates(candidates)

		device = next(self.policy.parameters()).device
		generator = None if seed is None else torch.Generator(device).manual_seed(seed)

		if chords is None:

			def draw_whole(count: int) -> list[tuple[ChordEvent, ...]]:
				drafts = self.policy.generate(
					[melody] * count, temperature=config.temperature, generator=generator
				)
				return [merge_repeats(draft, melody) for draft in drafts]

			return _distinct(draw_whole, candidates, config.max_duplicates)

		chords = tuple(chords)
		self._check_chords(melody, chords)
		start, stop = _bar_span(melody, bars)
		before, after = _cut(chords, 0, start), _cut(chords, stop, melody.length)

		def draw_bars(count: int) -> list[tuple[ChordEvent, ...]]:
			middles = self._refill(melody, before, after, start, stop, count, config, generator)
			return [merge_repeats(before + middle + after, melody) for middle in middles]

		results = _distinct(draw_bars, candidates, config.max_duplicates)
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
		config = replace(self.generation_config, **overrides)
		HarmonyRL._check_candidates(candidates)

		if not 0 <= overlap <= MAX_OVERLAP:
			raise ValueError(f"overlap must be 0-{MAX_OVERLAP}, got {overlap}")

		counts = [_bar_count(part.melody) for part in sheet.parts]
		if bars is not None and (bars.step != 1 or not 0 <= bars.start < bars.stop <= sum(counts)):
			raise ValueError(
				f"bars must be a nonempty range with step 1 within 0..{sum(counts)}, got {bars}"
			)

		attempts = 0

		def draw(count: int) -> list[LeadSheet]:
			nonlocal attempts
			drawn = []
			for index in range(attempts, attempts + count):
				draws = random.Random(f"{seed}/{index}") if seed is not None else None
				drawn.append(self._attempt(sheet, counts, bars, overlap, draws, overrides))

			attempts += count
			return drawn

		return _distinct(draw, candidates, config.max_duplicates)

	def _attempt(
		self,
		sheet: LeadSheet,
		counts: list[int],
		bars: range | None,
		overlap: int,
		draws: random.Random | None,
		overrides: dict[str, Any],
	) -> LeadSheet:
		"""1 harmonization of `sheet`, whole or only `bars`."""
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

		return replace(sheet, parts=tuple(parts))

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
			current = _splice(current, window_start, window_end, drawn)
			position = stop

		return merge_repeats(current, melody)

	def _refill(
		self,
		melody: Melody,
		before: tuple[ChordEvent, ...],
		after: tuple[ChordEvent, ...],
		start: int,
		stop: int,
		count: int,
		config: GenerationConfig,
		generator: torch.Generator | None,
	) -> list[tuple[ChordEvent, ...]]:
		"""`count` drafts of the chords from `start` to `stop`, each picked from its own pool."""
		size = count * config.pool
		drafts = self.policy.generate(
			[melody] * size,
			prefixes=[before] * size,
			temperature=config.temperature,
			generator=generator,
		)
		middles = [_cut(draft, start, stop) for draft in drafts]
		weights = torch.zeros(size, device=next(self.policy.parameters()).device)

		# The policy reads left to right, so the chords after the bars weigh each draft instead.
		if after:
			distinct = list(dict.fromkeys(middles))
			fit = self._log_probs(melody, [before + middle + after for middle in distinct])
			fit -= self._log_probs(melody, [before + middle for middle in distinct])
			index = {middle: i for i, middle in enumerate(distinct)}
			weights = fit[[index[middle] for middle in middles]]

		picks = torch.multinomial(
			weights.view(count, config.pool).softmax(1), 1, generator=generator
		).squeeze(1)

		return [middles[row * config.pool + pick] for row, pick in enumerate(picks.tolist())]

	def _check_chords(self, melody: Melody, chords: tuple[ChordEvent, ...]) -> None:
		if sum(event.duration for event in chords) != melody.length:
			raise ValueError(f"chords must cover the melody's {melody.length} 8th notes exactly")

		try:
			prefix_targets(melody, chords, restrict_no_chord=self.policy.config.restrict_no_chord)
		except ValueError as error:
			raise ValueError(f"chords the policy cannot follow: {error}") from error

	@staticmethod
	def _check_length(melody: Melody) -> None:
		bars = ceil(melody.length / melody.meter.bar_eighths)
		if bars > MAX_BARS:
			raise ValueError(f"the melody has {bars} bars, more than {MAX_BARS}")

	@staticmethod
	def _check_candidates(candidates: int) -> None:
		if not 1 <= candidates <= MAX_CANDIDATES:
			raise ValueError(f"candidates must be 1-{MAX_CANDIDATES}, got {candidates}")

	@torch.no_grad()
	def _log_probs(
		self, melody: Melody, harmonizations: Sequence[tuple[ChordEvent, ...]]
	) -> torch.Tensor:
		features = melody_features(melody)
		restrict = self.policy.config.restrict_no_chord
		batch = collate(
			[
				(features, prefix_targets(melody, h, restrict_no_chord=restrict))
				for h in harmonizations
			]
		)
		device = next(self.policy.parameters()).device
		return self.policy(batch.to(device)).log_prob.sum(dim=1)


def _distinct[T: Hashable](
	draw: Callable[[int], list[T]], candidates: int, max_duplicates: int
) -> list[T]:
	"""Results of `draw` until `candidates` differ or `max_duplicates` repeat earlier ones."""
	found: dict[T, None] = {}
	duplicates = 0

	while len(found) < candidates and duplicates < max_duplicates:
		for result in draw(candidates - len(found)):
			if result in found:
				duplicates += 1
			else:
				found[result] = None

	return list(found)


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


def _splice(
	events: Sequence[ChordEvent], start: int, end: int, middle: Sequence[ChordEvent]
) -> tuple[ChordEvent, ...]:
	"""`events` with `start` to `end` replaced by `middle`, a chord cut at an edge held again."""
	length = sum(event.duration for event in events)
	onsets = set(accumulate((event.duration for event in events), initial=0))
	spliced: list[ChordEvent] = []
	position = 0

	for event in (*_cut(events, 0, start), *middle, *_cut(events, end, length)):
		if (
			spliced
			and position in (start, end)
			and position not in onsets
			and event.chord == spliced[-1].chord
		):
			spliced[-1] = ChordEvent(event.chord, spliced[-1].duration + event.duration)
		else:
			spliced.append(event)

		position += event.duration

	return tuple(spliced)


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
