"""ChordCritic: a bidirectional Transformer over frames that judges a whole harmonization."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import batched

import torch
from torch import nn

from harmonyrl.schema import HEAD_VALUES, ChordEvent, Melody

from ..attention import Block
from ..features import (
	BAR_POSITIONS,
	BEAT_STRENGTHS,
	MAX_BEATS_PER_BAR,
	MODES,
	OCTAVES,
	SECTIONS,
)
from ..hub import StrictModelHubMixin, head_values_json
from .batch import CriticBatch, collate
from .features import CHORD_HEADS, critic_inputs


@dataclass(frozen=True)
class CriticConfig:
	"""Structure of `ChordCritic`, which `save_pretrained` stores as `config.json`.

	`head_values` is the vocabulary the chord inputs index, checked against the code that loads it.
	"""

	dim: int = 192
	attention_heads: int = 4
	layers: int = 4
	ff_mult: int = 4
	dropout: float = 0.1
	head_values: dict[str, list] = field(default_factory=head_values_json)


class ChordCritic(nn.Module, StrictModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""Logit that a harmonization of a melody is a human accompaniment."""

	def __init__(self, config: CriticConfig) -> None:
		super().__init__()

		if config.head_values != head_values_json():
			raise ValueError("head_values differ from the HEAD_VALUES of this code")

		self.config = config
		dim = config.dim

		self.pitch_classes = nn.Linear(12, dim)
		self.frame_embeddings = nn.ModuleDict(
			{
				"onset": nn.Embedding(2, dim),
				"octave": nn.Embedding(OCTAVES + 1, dim),
				"bar_position": nn.Embedding(BAR_POSITIONS, dim),
				"beat_strength": nn.Embedding(BEAT_STRENGTHS, dim),
				"mode": nn.Embedding(len(MODES), dim),
				"key_start": nn.Embedding(2, dim),
			}
		)
		self.melody_embeddings = nn.ModuleDict(
			{
				"beats_per_bar": nn.Embedding(MAX_BEATS_PER_BAR + 1, dim),
				"beat_unit": nn.Embedding(2, dim),
				"section": nn.Embedding(len(SECTIONS) + 1, dim),
			}
		)
		self.chord_values = nn.ModuleList(
			nn.Embedding(len(HEAD_VALUES[head]) + 1, dim) for head in CHORD_HEADS
		)
		self.chord_tones = nn.Linear(12, dim)
		self.chord_bass = nn.Embedding(13, dim)
		self.chord_onset = nn.Embedding(2, dim)

		self.blocks = nn.ModuleList(
			Block(dim, config.attention_heads, config.ff_mult, config.dropout, cross=False)
			for _ in range(config.layers)
		)
		self.norm = nn.LayerNorm(dim)
		self.output = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

	def forward(self, batch: CriticBatch) -> torch.Tensor:
		"""Logits (batch,)."""
		x = self.pitch_classes(batch.pitch_classes) + self.chord_tones(batch.chord_tones)

		for name, embedding in self.frame_embeddings.items():
			x = x + embedding(getattr(batch, name).long())

		for name, embedding in self.melody_embeddings.items():
			x = x + embedding(getattr(batch, name))[:, None]

		for index, embedding in enumerate(self.chord_values):
			x = x + embedding(batch.chord_values[..., index])

		x = x + self.chord_bass(batch.chord_bass) + self.chord_onset(batch.chord_onset.long())

		size, frames = batch.frame_mask.shape
		times = torch.arange(frames, device=x.device).expand(size, frames)
		mask = batch.frame_mask[:, None].expand(size, frames, frames)

		for block in self.blocks:
			x = block(x, times, mask)

		weights = batch.frame_mask[..., None].to(x.dtype)
		pooled = (self.norm(x) * weights).sum(dim=1) / weights.sum(dim=1)

		return self.output(pooled).squeeze(-1)

	@torch.no_grad()
	def score(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int = 64,
	) -> list[float]:
		"""Logit of each harmonization, scored `batch_size` at a time among similar lengths.

		The critic should be run in eval mode.
		"""
		device = next(self.parameters()).device
		order = sorted(range(len(melodies)), key=lambda i: melodies[i].length)
		logits = [0.0] * len(melodies)

		for chunk in batched(order, batch_size, strict=False):
			items = [critic_inputs(melodies[i], harmonizations[i]) for i in chunk]
			batch = collate(items, [0.0] * len(items)).to(device)

			for i, logit in zip(chunk, self(batch).float().tolist(), strict=True):
				logits[i] = logit

		return logits
