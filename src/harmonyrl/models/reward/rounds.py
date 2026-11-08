"""Rounds of preference rankings: continuations of the human chords of 1 melody, to be ranked."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations

import numpy as np
import torch

from harmonyrl.metrics import chord_spans, out_of_key_share
from harmonyrl.render import DEFAULT_TEMPO
from harmonyrl.schema import ChordEvent, Melody, Segment, merge_repeats

from ..policy import ChordPolicy
from .model import ChordReward


@dataclass(frozen=True)
class RankingGroup:
	"""Continuations, the `items`, of the first `prefix` human events of `segment`."""

	segment: Segment
	prefix: int
	items: tuple[tuple[ChordEvent, ...], ...]


def seconds(melody: Melody) -> float:
	"""Length of `melody` in seconds at its tempo, or at `DEFAULT_TEMPO` when that is unknown."""
	tempo = DEFAULT_TEMPO if melody.tempo is None else melody.tempo
	quarter_bpm = Fraction(tempo * (3 if melody.meter.beat_unit == 8 else 2), 2)

	return float(melody.length * Fraction(30) / quarter_bpm)


def prefix_split(segment: Segment) -> int | None:
	"""Number of human events up to the event boundary on a barline nearest the middle of the
	melody, with 2 bars or more on each side, or None when there is none."""
	melody = segment.melody
	bar = melody.meter.bar_eighths
	middle = melody.length / 2

	best: tuple[int, int] | None = None
	position = 0

	for count, event in enumerate(segment.events, start=1):
		position += event.duration
		if (
			position % bar == 0
			and 2 * bar <= position <= melody.length - 2 * bar
			and (best is None or abs(position - middle) < abs(best[1] - middle))
		):
			best = (count, position)

	return None if best is None else best[0]


def sounding_difference(melody: Melody, a: Sequence[ChordEvent], b: Sequence[ChordEvent]) -> int:
	"""8th notes where the chords of `a` and `b` sound different pitch classes or bass."""

	def frames(events: Sequence[ChordEvent]) -> list[tuple[int | None, frozenset[int]]]:
		sounding: list[tuple[int | None, frozenset[int]]] = []

		for span in chord_spans(melody, merge_repeats(events, melody)):
			sounding += [(span.bass, span.pitch_classes)] * (span.end - span.start)

		return sounding

	return sum(x != y for x, y in zip(frames(a), frames(b), strict=True))


def ranking_groups(
	policy: ChordPolicy,
	segments: Sequence[Segment],
	count: int,
	items: int,
	temperature: float,
	min_bars_apart: int,
	generator: torch.Generator,
) -> list[RankingGroup]:
	"""Up to `count` groups from `segments` in order, each with `items` continuations of the prefix
	of its segment that sound different 2 by 2 for `min_bars_apart` bars or more."""
	groups: list[RankingGroup] = []

	for segment in segments:
		if len(groups) == count:
			break

		prefix = prefix_split(segment)
		if prefix is None:
			continue

		continuations = policy.generate(
			[segment.melody] * items,
			prefixes=[segment.events[:prefix]] * items,
			temperature=temperature,
			generator=generator,
		)
		if _apart(segment.melody, continuations, min_bars_apart):
			groups.append(RankingGroup(segment, prefix, tuple(continuations)))

	return groups


def head_disagreement(reward: ChordReward, groups: Sequence[RankingGroup]) -> list[float]:
	"""How much the heads of `reward` disagree on the order of the items of each group: over every
	2 items, 1 - |2p - 1| for the share p of heads that score the 1st higher, averaged."""
	melodies = [g.segment.melody for g in groups for _ in g.items]
	scores = reward.head_scores(melodies, [items for g in groups for items in g.items])
	disagreement: list[float] = []
	start = 0

	for group in groups:
		group_scores = scores[start : start + len(group.items)]
		start += len(group.items)
		shares = [
			float((group_scores[i] > group_scores[j]).mean())
			for i, j in combinations(range(len(group.items)), 2)
		]
		disagreement.append(float(np.mean([1 - abs(2 * p - 1) for p in shares])))

	return disagreement


def best_of_group(
	policy: ChordPolicy,
	reward: ChordReward,
	group: RankingGroup,
	candidates: int,
	temperature: float,
	min_bars_apart: int,
	generator: torch.Generator,
) -> RankingGroup | None:
	"""`group` with its 1st item replaced by the best of `candidates` new continuations under
	`reward`, or None when that best sounds too close to another item."""
	melody = group.segment.melody
	drawn = policy.generate(
		[melody] * candidates,
		prefixes=[group.segment.events[: group.prefix]] * candidates,
		temperature=temperature,
		generator=generator,
	)
	best = drawn[int(np.argmax(reward.score([melody] * candidates, drawn)))]
	items = (best, *group.items[1:])

	return replace(group, items=items) if _apart(melody, items, min_bars_apart) else None


def color_groups(
	policy: ChordPolicy,
	segments: Sequence[Segment],
	count: int,
	betas: Sequence[float],
	candidates: int,
	temperature: float,
	min_bars_apart: int,
	generator: torch.Generator,
) -> list[RankingGroup]:
	"""Up to `count` groups from `segments` in order."""
	groups: list[RankingGroup] = []

	for segment in segments:
		if len(groups) == count:
			break

		melody = segment.melody
		plain, *pool = policy.generate(
			[melody] * (candidates + 1),
			temperature=temperature,
			generator=generator,
		)
		shares = torch.tensor(
			[out_of_key_share(melody, events) for events in pool], device=generator.device
		)
		items = [plain]

		for beta in betas:
			index = torch.multinomial((shares / beta).softmax(dim=0), 1, generator=generator)
			items.append(pool[int(index)])

		colored = max(out_of_key_share(melody, events) for events in items[1:])
		if colored > out_of_key_share(melody, plain) and _apart(melody, items, min_bars_apart):
			groups.append(RankingGroup(segment, 0, tuple(items)))

	return groups


def policy_groups(
	policies: Sequence[tuple[ChordPolicy, float]],
	segments: Sequence[Segment],
	count: int,
	min_bars_apart: int,
	generator: torch.Generator,
) -> list[RankingGroup]:
	"""Up to `count` groups from `segments` in order, each with 1 harmonization by every policy at
	its temperature, that sound different 2 by 2 for `min_bars_apart` bars or more."""
	groups: list[RankingGroup] = []

	for segment in segments:
		if len(groups) == count:
			break

		items = tuple(
			policy.generate([segment.melody], temperature=temperature, generator=generator)[0]
			for policy, temperature in policies
		)

		if _apart(segment.melody, items, min_bars_apart):
			groups.append(RankingGroup(segment, 0, items))

	return groups


def _apart(melody: Melody, items: Sequence[Sequence[ChordEvent]], min_bars_apart: int) -> bool:
	apart = min_bars_apart * melody.meter.bar_eighths
	return all(sounding_difference(melody, a, b) >= apart for a, b in combinations(items, 2))
