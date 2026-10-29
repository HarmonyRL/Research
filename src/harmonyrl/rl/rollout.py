"""Rollouts: policy samples for a batch of segments, with the log π that RL learns from."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from harmonyrl.models import (
	Batch,
	ChordCritic,
	ChordPolicy,
	collate,
	event_targets,
	melody_features,
)
from harmonyrl.schema import ChordEvent, Segment

from .reward import terminal_rewards


@dataclass(frozen=True)
class Rollout:
	"""1 sampled harmonization per segment, and what was computed from it.

	`samples` keep the id, split, and melody of their segments. `batch` holds their targets
	under the masks they were sampled under. `log_prob` and `reference_log_prob` are log π
	of each event (segments, events) under the policy and the reference, 0 on padded events,
	and `rewards` (segments,) are terminal rewards. Tensors are on the policy's device.
	"""

	samples: tuple[Segment, ...]
	batch: Batch
	log_prob: torch.Tensor
	reference_log_prob: torch.Tensor
	rewards: torch.Tensor


def policy_from_reference(reference: ChordPolicy) -> ChordPolicy:
	"""The policy RL starts from: `reference` without dropout, keeping no chord to a pickup.

	Without dropout, log π of an action is the same at every forward pass, so the log π of a
	rollout is the one its gradient step sees.
	"""
	policy = ChordPolicy(replace(reference.config, dropout=0.0, restrict_no_chord=True))
	policy.load_state_dict(reference.state_dict())

	return policy


@torch.no_grad()
def rollout(
	policy: ChordPolicy,
	reference: ChordPolicy,
	critic: ChordCritic,
	segments: Sequence[Segment],
	*,
	clip: float,
	generator: torch.Generator | None = None,
) -> Rollout:
	"""Sample each melody of `segments` from `policy` at temperature 1 and score the samples.

	The reference is evaluated under the masks of the policy, so both log π are over the same
	actions. All 3 models should be in eval mode and on 1 device.
	"""
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
		rewards=terminal_rewards(critic, melodies, events, clip).to(device),
	)


def rollout_stats(rollout: Rollout) -> dict[str, float]:
	"""Means over the samples of `rollout`, for watching training.

	`kl` sums log π - log π_ref over the events of a sample, `nll` is -log π per event,
	and `no_chord` the share of 8th notes under no chord. `longest_hold` is the longest run of
	1 chord, in bars, and `longest_alternation` the most events in a row that swing between 2 chords
	(0 when none do)."""
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
