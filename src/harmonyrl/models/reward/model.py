"""ChordReward: linear heads fit to preferences over frozen features of a harmonization."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from torch import nn

from harmonyrl.schema import ChordEvent, Melody

from ..hub import StrictModelHubMixin
from ..policy import ChordPolicy
from .extractors import LikelihoodFeatures, MusicbertFeatures, PolicyFeatures


@dataclass(frozen=True)
class RewardConfig:
	"""Structure of `ChordReward`, which `save_pretrained` stores as `config.json`."""

	features: str = "likelihood"
	policy: str = "runs/pretrain/final/best"
	backbone: str = "data/pretrained/musicbert"
	tokenizer: str = "data/pretrained/miditok-REMI"
	max_tokens: int = 1024
	heads: int = 5


class ChordReward(nn.Module, StrictModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""Preference score of a harmonization of a melody: a linear head over its standardized frozen
	features, with bootstrap heads beside it to measure disagreement."""

	feature_mean: torch.Tensor
	feature_std: torch.Tensor

	def __init__(self, config: RewardConfig) -> None:
		super().__init__()

		self.config = config
		self.features = _features(config, pretrained=False)
		self.register_buffer("feature_mean", torch.zeros(self.features.dim))
		self.register_buffer("feature_std", torch.ones(self.features.dim))
		self.heads = nn.Linear(self.features.dim, config.heads)

	@classmethod
	def from_backbone(cls, config: RewardConfig) -> Self:
		"""A reward with the pretrained features and untrained heads."""
		reward = cls(config)
		reward.features = _features(config, pretrained=True)
		return reward

	def score_features(self, features: torch.Tensor) -> torch.Tensor:
		"""Scores of every head (batch, heads) for features from `features`."""
		return self.heads((features - self.feature_mean) / self.feature_std)

	@torch.no_grad()
	def head_scores(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> np.ndarray:
		"""Scores of every head (harmonizations, heads)."""
		features = self.features(melodies, harmonizations, batch_size)
		return self.score_features(features).float().cpu().numpy()

	def score(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> list[float]:
		"""Score of each harmonization under the 1st head, the one fit to every comparison."""
		return self.head_scores(melodies, harmonizations, batch_size)[:, 0].tolist()


def _features(
	config: RewardConfig, pretrained: bool
) -> LikelihoodFeatures | PolicyFeatures | MusicbertFeatures:
	if config.features == "likelihood":
		return LikelihoodFeatures(ChordPolicy.from_pretrained(config.policy))
	if config.features == "policy":
		return PolicyFeatures(ChordPolicy.from_pretrained(config.policy))
	if config.features == "musicbert":
		return MusicbertFeatures(config.backbone, config.tokenizer, config.max_tokens, pretrained)

	raise ValueError(f"features must be policy or musicbert, got {config.features!r}")
