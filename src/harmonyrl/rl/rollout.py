"""Rollouts: policy samples for a batch of segments, with the log π that RL learns from."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import torch
from torch import nn

from harmonyrl.models import (
	Batch,
	ChordPolicy,
	collate,
	event_targets,
	melody_features,
)
from harmonyrl.schema import ChordEvent, Segment


@dataclass(frozen=True)
class Rollout:
	"""1 sampled harmonization per segment, and what was computed from it."""

	samples: tuple[Segment, ...]
	batch: Batch
	log_prob: torch.Tensor
	reference_log_prob: torch.Tensor
	rewards: torch.Tensor


def policy_from_reference(reference: ChordPolicy) -> ChordPolicy:
	"""The policy RL starts from: `reference` without dropout, keeping no chord to a pickup."""
	policy = ChordPolicy(replace(reference.config, dropout=0.0, restrict_no_chord=True))
	policy.load_state_dict(reference.state_dict())

	return policy


@torch.no_grad()
def sharpen(reference: ChordPolicy, temperature: float) -> None:
	"""Make `reference` at temperature 1 sample as it did at `temperature`."""
	for output in reference.outputs:
		layer = cast(nn.Linear, cast(nn.Sequential, output)[-1])
		layer.weight.div_(temperature)
		layer.bias.div_(temperature)


@torch.no_grad()
def rollout(
	policy: ChordPolicy,
	reference: ChordPolicy,
	reward: Callable[[Sequence[Segment], Sequence[Segment]], torch.Tensor],
	segments: Sequence[Segment],
	*,
	generator: torch.Generator | None = None,
) -> Rollout:
	"""Sample each melody of `segments` from `policy` at temperature 1 and score the samples."""
	if not policy.config.restrict_no_chord:
		raise ValueError("the policy must restrict no chord")

	melodies = [s.melody for s in segments]
	events = policy.generate(melodies, generator=generator)
	samples = tuple(
		Segment(s.id, s.split, s.melody, e) for s, e in zip(segments, events, strict=True)
	)

	device = policy.not_predicted.device
	items = [(melody_features(s.melody), event_targets(s, restrict_no_chord=True)) for s in samples]
	batch = collate(items).to(device)

	return Rollout(
		samples=samples,
		batch=batch,
		log_prob=policy(batch).log_prob,
		reference_log_prob=reference(batch).log_prob,
		rewards=reward(samples, segments).to(device),
	)


def rollout_stats(rollout: Rollout) -> dict[str, float]:
	"""Means over the samples of `rollout`, for watching training."""
	mask = rollout.batch.event_mask
	samples = rollout.samples
	no_chord = sum(e.duration for s in samples for e in s.events if e.chord is None)

	return {
		"reward": rollout.rewards.mean().item(),
		"kl": (rollout.log_prob - rollout.reference_log_prob).sum(dim=1).mean().item(),
		"nll": -rollout.log_prob[mask].mean().item(),
		"events": mask.sum(dim=1).float().mean().item(),
		"no_chord": no_chord / sum(s.melody.length for s in samples),
		"longest_hold": float(np.mean([_longest_hold(s) for s in samples])),
		"longest_alternation": float(np.mean([_longest_alternation(s.events) for s in samples])),
	}


def _longest_hold(segment: Segment) -> float:
	longest = run = 0
	previous = None

	for event in segment.events:
		if event.chord is None:
			run = 0
		elif event.chord == previous:
			run += event.duration
		else:
			run = event.duration

		longest = max(longest, run)
		previous = event.chord

	return longest / segment.melody.meter.bar_eighths


def _longest_alternation(events: Sequence[ChordEvent]) -> int:
	chords = [e.chord for e in events]
	longest = run = 0

	for i in range(2, len(chords)):
		swings = (
			chords[i] is not None
			and chords[i - 1] is not None
			and chords[i] == chords[i - 2]
			and chords[i] != chords[i - 1]
		)
		run = (run + 1 if run else 3) if swings else 0
		longest = max(longest, run)

	return longest
