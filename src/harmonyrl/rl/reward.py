"""Terminal reward: the clipped score of the fixed preference reward."""

from collections.abc import Sequence

import torch

from harmonyrl.models import ChordReward
from harmonyrl.schema import ChordEvent, Melody


def terminal_rewards(
	reward: ChordReward,
	melodies: Sequence[Melody],
	harmonizations: Sequence[Sequence[ChordEvent]],
	clip: float,
) -> torch.Tensor:
	"""Reward of each harmonization, its preference score clipped to [-`clip`, `clip`].

	The reward model should be run in eval mode. Rewards are on its device.
	"""
	if clip <= 0:
		raise ValueError(f"clip must be positive, got {clip}")

	device = next(reward.parameters()).device
	scores = torch.tensor(reward.score(melodies, harmonizations), device=device)

	return scores.clamp(-clip, clip)
