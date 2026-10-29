"""Rollouts: policy samples for a batch of segments, with the log π that PPO and GDC learn from."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from harmonyrl.models import (
	Batch,
	ChordCritic,
	ChordPolicy,
	collate,
	event_targets,
	melody_features,
)
from harmonyrl.schema import Segment

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
