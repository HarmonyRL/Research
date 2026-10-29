"""Terminal reward: the clipped logit of the fixed structure critic."""

from collections.abc import Sequence

import torch

from harmonyrl.models import ChordCritic
from harmonyrl.schema import ChordEvent, Melody


def terminal_rewards(
	critic: ChordCritic,
	melodies: Sequence[Melody],
	harmonizations: Sequence[Sequence[ChordEvent]],
	clip: float,
) -> torch.Tensor:
	"""Reward of each harmonization, its critic logit clipped to [-`clip`, `clip`].

	The critic should be run in eval mode. Rewards are on the critic's device.
	"""
	if clip <= 0:
		raise ValueError(f"clip must be positive, got {clip}")

	device = next(critic.parameters()).device
	logits = torch.tensor(critic.score(melodies, harmonizations), device=device)

	return logits.clamp(-clip, clip)
