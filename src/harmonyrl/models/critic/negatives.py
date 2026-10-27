"""Negative examples for the critic: policy samples, and human accompaniments broken in 1 place.

A perturbation changes 1 spot and keeps every other event as it was. Where it changes which chord
sounds, the melody spends no less time on chord tones than before, so that consonance cannot tell
a perturbed accompaniment from a human one and the critic has to judge structure.
"""

from collections.abc import Callable, Sequence

import numpy as np

from harmonyrl.schema import ChordEvent, Segment

from .harmony import old_key_chord, phrase_end, repeat, swap
from .pool import ChordPool
from .spans import Span, projected, to_events
from .timing import added_change, missed_change, moved_change

type Perturbation = Callable[[Segment, ChordPool, np.random.Generator], list[Span] | None]

PERTURBATIONS: dict[str, Perturbation] = {
	"old_key_chord": old_key_chord,
	"repeat": repeat,
	"swap": swap,
	"moved_change": moved_change,
	"missed_change": missed_change,
	"added_change": added_change,
	"phrase_end": phrase_end,
}
"""Perturbations by name, the ones that apply to the fewest segments first."""


def perturb(
	segment: Segment, kind: str, pool: ChordPool, rng: np.random.Generator
) -> tuple[ChordEvent, ...] | None:
	"""The events of `segment` broken by perturbation `kind`, or None where it cannot apply."""
	spans = PERTURBATIONS[kind](segment, pool, rng)
	return None if spans is None else to_events(spans)


def negatives(
	segments: Sequence[Segment],
	samples: Sequence[Sequence[ChordEvent]],
	pool: ChordPool,
	rng: np.random.Generator,
) -> tuple[list[Sequence[ChordEvent]], list[str]]:
	"""1 negative per segment, and its kind: `"sample"` or a perturbation name.

	Half the segments take their policy sample from `samples`. The other half share the
	perturbations equally, in the order of `PERTURBATIONS`. Segments left once a rare kind runs
	short take random kinds, and so does a segment whose sample equals its human accompaniment,
	which would otherwise be a positive labeled negative. A segment that no perturbation fits keeps
	its sample.
	"""
	order = rng.permutation(len(segments)).tolist()
	events: list[Sequence[ChordEvent]] = list(samples)
	kinds = ["sample"] * len(segments)
	remaining = order[: len(order) // 2]
	quota = len(remaining) // len(PERTURBATIONS)

	for kind in PERTURBATIONS:
		filled = 0

		for index in list(remaining):
			if filled == quota:
				break

			broken = perturb(segments[index], kind, pool, rng)
			if broken is not None:
				events[index], kinds[index] = broken, kind
				remaining.remove(index)
				filled += 1

	copies = [
		index
		for index in order[len(order) // 2 :]
		if projected(segments[index], samples[index])
		== projected(segments[index], segments[index].events)
	]

	for index in [*remaining, *copies]:
		for kind in rng.permutation(list(PERTURBATIONS)).tolist():
			broken = perturb(segments[index], kind, pool, rng)
			if broken is not None:
				events[index], kinds[index] = broken, kind
				break

	return events, kinds
