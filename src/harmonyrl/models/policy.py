"""ChordPolicy: a melody encoder and an autoregressive decoder over chord events."""

import json
from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from harmonyrl.schema import HEAD_VALUES, MAX_DURATION

from .attention import Block
from .batch import Batch
from .features import (
	BAR_POSITIONS,
	BEAT_STRENGTHS,
	IGNORE,
	MAX_BEATS_PER_BAR,
	MODES,
	OCTAVES,
	SECTIONS,
)


def head_values_json() -> dict[str, list]:
	"""`HEAD_VALUES` as a checkpoint config stores it, with modes as strings and tuples as lists."""
	return json.loads(json.dumps(HEAD_VALUES))


@dataclass(frozen=True)
class PolicyOutput:
	"""Masked logits of each head (batch, events, values), and log π of each event (batch, events).

	`log_prob` sums the log probabilities of the target heads, so it is 0 on padded events.
	"""

	logits: dict[str, torch.Tensor]
	log_prob: torch.Tensor


class ChordPolicy(nn.Module, PyTorchModelHubMixin, license="cc-by-nc-sa-3.0"):
	"""π(chord events | melody), factorized over events and over the heads of each event.

	Heads are predicted in `HEAD_VALUES` order, each conditioned on the heads before it.
	The structure depends only on the constructor arguments, which `save_pretrained` stores.
	"""

	not_predicted: torch.Tensor
	"""Per head, the value index that stands for a head not predicted."""

	def __init__(
		self,
		head_values: dict[str, list],
		dim: int = 256,
		attention_heads: int = 4,
		encoder_layers: int = 4,
		decoder_layers: int = 4,
		ff_mult: int = 4,
		dropout: float = 0.1,
	) -> None:
		super().__init__()

		if head_values != head_values_json():
			raise ValueError("head_values differ from the HEAD_VALUES of this code")

		sizes = {head: len(values) for head, values in HEAD_VALUES.items()}
		self.register_buffer("not_predicted", torch.tensor(list(sizes.values())), persistent=False)

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
		self.encoder = nn.ModuleList(
			Block(dim, attention_heads, ff_mult, dropout, cross=False)
			for _ in range(encoder_layers)
		)
		self.encoder_norm = nn.LayerNorm(dim)

		self.previous = nn.ModuleList(nn.Embedding(size + 1, dim) for size in sizes.values())
		self.remaining = nn.Embedding(MAX_DURATION + 1, dim)
		self.decoder = nn.ModuleList(
			Block(dim, attention_heads, ff_mult, dropout, cross=True) for _ in range(decoder_layers)
		)
		self.decoder_norm = nn.LayerNorm(dim)

		self.outputs = nn.ModuleList(
			nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, size))
			for size in sizes.values()
		)
		self.chosen = nn.ModuleList(
			nn.Embedding(size + 1, dim) for size in list(sizes.values())[:-1]
		)

	def encode(self, batch: Batch) -> torch.Tensor:
		"""Frame states (batch, frames, dim)."""
		x = self.pitch_classes(batch.pitch_classes)
		for name, embedding in self.frame_embeddings.items():
			x = x + embedding(getattr(batch, name).long())
		for name, embedding in self.melody_embeddings.items():
			x = x + embedding(getattr(batch, name))[:, None]

		size, frames = batch.frame_mask.shape
		times = torch.arange(frames, device=x.device).expand(size, frames)
		mask = batch.frame_mask[:, None].expand(size, frames, frames)

		for block in self.encoder:
			x = block(x, times, mask)

		return self.encoder_norm(x)

	def forward(self, batch: Batch) -> PolicyOutput:
		"""Logits and log π of the target events in `batch`, each event given the ones before."""
		frames = self.encode(batch)
		size, events, _ = batch.values.shape

		# A head that is not predicted, and every head before the 1st event, gets its own value.
		values = torch.where(batch.values == IGNORE, self.not_predicted, batch.values)
		previous = torch.cat((self.not_predicted.expand(size, 1, -1), values[:, :-1]), dim=1)
		x = frames.gather(1, batch.starts[..., None].expand(-1, -1, frames.shape[-1]))

		for index, embedding in enumerate(self.previous):
			x = x + embedding(previous[..., index])

		lengths = batch.frame_mask.sum(dim=1, keepdim=True)
		x = x + self.remaining((lengths - batch.starts).clamp(max=MAX_DURATION))

		causal = torch.ones(events, events, dtype=torch.bool, device=x.device).tril()
		self_mask = causal & batch.event_mask[:, None]
		cross_mask = batch.frame_mask[:, None].expand(size, events, -1)
		frame_times = torch.arange(frames.shape[1], device=x.device).expand(size, -1)

		for block in self.decoder:
			x = block(x, batch.starts, self_mask, frames, frame_times, cross_mask)
		x = self.decoder_norm(x)

		logits: dict[str, torch.Tensor] = {}
		log_prob = torch.zeros(size, events, device=x.device)

		for index, (head, output) in enumerate(zip(HEAD_VALUES, self.outputs, strict=True)):
			logits[head] = output(x).float().masked_fill(~batch.masks[head], float("-inf"))
			target = batch.values[..., index]
			chosen = logits[head].log_softmax(-1).gather(-1, target.clamp(min=0)[..., None])
			log_prob = log_prob + torch.where(target == IGNORE, 0.0, chosen.squeeze(-1))

			if index < len(self.chosen):
				x = x + self.chosen[index](values[..., index])

		return PolicyOutput(logits=logits, log_prob=log_prob)
