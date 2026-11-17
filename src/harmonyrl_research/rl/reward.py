"""Terminal rewards: the share of chords outside the key, or the score of a preference reward."""

from collections.abc import Sequence

import torch

from harmonyrl_research.metrics import held_share, out_of_key_share
from harmonyrl_research.models import ChordReward
from harmonyrl_research.schema import ChordEvent, Melody, Segment


def color_rewards(samples: Sequence[Segment], hold_penalty: float) -> torch.Tensor:
	"""Reward of each sample: the share of it played by chords outside the key."""
	return torch.tensor(
		[
			out_of_key_share(s.melody, s.events) - hold_penalty * held_share(s.melody, s.events)
			for s in samples
		],
		dtype=torch.float32,
	)


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
