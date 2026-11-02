"""ChordCritic: a pretrained symbolic-music encoder that judges a whole harmonization."""

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
from .batch import CriticBatch, collate
from .tokens import token_ids


@dataclass(frozen=True)
class CriticConfig:
	"""Structure of `ChordCritic`, which `save_pretrained` stores as `config.json`."""

	backbone: str = "data/pretrained/musicbert"
	tokenizer: str = "data/pretrained/miditok-REMI"
	max_tokens: int = 1024


class ChordCritic(nn.Module, StrictModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""Logit that a harmonization of a melody is a human accompaniment."""

	def __init__(self, config: CriticConfig) -> None:
		super().__init__()

		self.config = config
		self.tokenizer = MusicTokenizer.from_pretrained(config.tokenizer)

		backbone = BertConfig.from_pretrained(config.backbone)
		self.encoder = BertModel(backbone, add_pooling_layer=False)
		self.output = nn.Linear(backbone.hidden_size, 1)

	@classmethod
	def from_backbone(cls, config: CriticConfig) -> Self:
		"""A critic with the pretrained encoder of `config.backbone` and an untrained output."""
		critic = cls(config)
		critic.encoder = BertModel.from_pretrained(config.backbone, add_pooling_layer=False)
		return critic

	def forward(self, batch: CriticBatch) -> torch.Tensor:
		"""Logits (batch,)."""
		states = self.encoder(
			input_ids=batch.input_ids, attention_mask=batch.attention_mask
		).last_hidden_state
		weights = batch.attention_mask[..., None].to(states.dtype)
		pooled = (states * weights).sum(dim=1) / weights.sum(dim=1)

		return self.output(pooled).squeeze(-1)

	def token_ids(self, melody: Melody, events: Sequence[ChordEvent]) -> np.ndarray:
		"""The input of 1 harmonization."""
		return token_ids(self.tokenizer, melody, events, self.config.max_tokens)

	@torch.no_grad()
	def score(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> list[float]:
		"""Logits of each harmonization, scored `batch_size` at a time among similar lengths.

		The critic should be run in eval mode.
		"""
		device = next(self.parameters()).device
		inputs = [self.token_ids(m, e) for m, e in zip(melodies, harmonizations, strict=True)]
		order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
		logits = [0.0] * len(inputs)

		for chunk in batched(order, batch_size, strict=False):
			batch = collate([inputs[i] for i in chunk], [0.0] * len(chunk)).to(device)

			for i, logit in zip(chunk, self(batch).float().tolist(), strict=True):
				logits[i] = logit

		return logits
