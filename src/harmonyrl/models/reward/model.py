"""ChordReward: a frozen pretrained symbolic-music encoder with linear heads fit to preferences."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Self

import numpy as np
import torch
from miditok import MusicTokenizer
from torch import nn
from transformers import BertConfig, BertModel

from harmonyrl.schema import ChordEvent, Melody

from ..hub import StrictModelHubMixin
from .batch import RewardBatch, collate
from .tokens import token_ids


@dataclass(frozen=True)
class RewardConfig:
	"""Structure of `ChordReward`, which `save_pretrained` stores as `config.json`."""

	backbone: str = "data/pretrained/musicbert"
	tokenizer: str = "data/pretrained/miditok-REMI"
	max_tokens: int = 1024
	heads: int = 5


class ChordReward(nn.Module, StrictModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""Preference scrore of a harmonization of a melody: the mean of linear heads over the
	standardized, mean-pooled states of a frozen encoder."""

	feature_mean: torch.Tensor
	feature_std: torch.Tensor

	def __init__(self, config: RewardConfig) -> None:
		super().__init__()

		self.config = config
		self.tokenizer = MusicTokenizer.from_pretrained(config.tokenizer)

		backbone = BertConfig.from_pretrained(config.backbone)
		self.encoder = BertModel(backbone, add_pooling_layer=False).requires_grad_(False)
		self.register_buffer("feature_mean", torch.zeros(backbone.hidden_size))
		self.register_buffer("feature_std", torch.ones(backbone.hidden_size))
		self.heads = nn.Linear(backbone.hidden_size, config.heads)

	@classmethod
	def from_backbone(cls, config: RewardConfig) -> Self:
		"""A reward with the pretrained encoder of `config.backbone` and untrained heads."""
		reward = cls(config)
		reward.encoder = BertModel.from_pretrained(
			config.backbone, add_pooling_layer=False
		).requires_grad_(False)

		return reward

	def embed(self, batch: RewardBatch) -> torch.Tensor:
		"""Mean-pooled encoder states (batch, hidden)."""
		states = self.encoder(
			input_ids=batch.input_ids, attention_mask=batch.attention_mask
		).last_hidden_state
		weights = batch.attention_mask[..., None].to(states.dtype)

		return (states * weights).sum(dim=1) / weights.sum(dim=1)

	def score_features(self, features: torch.Tensor) -> torch.Tensor:
		"""Scores of every head (batch, heads) for features from `embed`."""
		return self.heads((features - self.feature_mean) / self.feature_std)

	def forward(self, batch: RewardBatch) -> torch.Tensor:
		"""Scores of every head (batch, heads)."""
		return self.score_features(self.embed(batch))

	def token_ids(self, melody: Melody, events: Sequence[ChordEvent]) -> np.ndarray:
		"""The input of 1 harmonization."""
		return token_ids(self.tokenizer, melody, events, self.config.max_tokens)

	@torch.no_grad()
	def head_scores(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> np.ndarray:
		"""Scores of every head (harmonizations, heads), `batch_size` at a time among similar
		lengths. The reward should be run in eval mode."""
		device = next(self.parameters()).device
		inputs = [self.token_ids(m, e) for m, e in zip(melodies, harmonizations, strict=True)]
		order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]), reverse=True)
		scores = np.zeros((len(inputs), self.config.heads), dtype=np.float32)

		for chunk in batched(order, batch_size, strict=False):
			batch = collate([inputs[i] for i in chunk]).to(device)
			scores[list(chunk)] = self(batch).float().cpu().numpy()

		return scores

	def score(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> list[float]:
		"""Mean head score of each harmonization. The reward should be run in eval mode."""
		return self.head_scores(melodies, harmonizations, batch_size).mean(axis=1).tolist()
