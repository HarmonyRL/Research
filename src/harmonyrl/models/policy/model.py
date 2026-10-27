"""ChordPolicy: a melody encoder and an autoregressive decoder over chord events."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from harmonyrl.schema import HEAD_VALUES, MAX_DURATION, ChordEvent, Melody, RawChord, allowed

from ..attention import Block
from ..batch import Batch, collate
from ..features import (
	BAR_POSITIONS,
	BEAT_STRENGTHS,
	IGNORE,
	MAX_BEATS_PER_BAR,
	MODES,
	OCTAVES,
	SECTIONS,
	EventTargets,
	melody_features,
)


def _head_values_json() -> dict[str, list]:
	return json.loads(json.dumps(HEAD_VALUES))


@dataclass(frozen=True)
class PolicyConfig:
	"""Structure of `ChordPolicy`, which `save_pretrained` stores as `config.json`.

	`head_values` is the vocabulary the output heads index: `HEAD_VALUES` with modes as strings
	and tuples as lists, so that a checkpoint can be checked against the code that loads it.
	"""

	dim: int = 384
	attention_heads: int = 4
	encoder_layers: int = 4
	decoder_layers: int = 4
	ff_mult: int = 4
	dropout: float = 0.3
	head_values: dict[str, list] = field(default_factory=_head_values_json)


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
	"""

	not_predicted: torch.Tensor
	"""Per head, the value index that stands for a head not predicted."""

	def __init__(self, config: PolicyConfig) -> None:
		super().__init__()

		if config.head_values != _head_values_json():
			raise ValueError("head_values differ from the HEAD_VALUES of this code")

		self.config = config
		dim = config.dim
		attention_heads = config.attention_heads
		ff_mult = config.ff_mult
		dropout = config.dropout

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
			for _ in range(config.encoder_layers)
		)
		self.encoder_norm = nn.LayerNorm(dim)

		self.previous = nn.ModuleList(nn.Embedding(size + 1, dim) for size in sizes.values())
		self.remaining = nn.Embedding(MAX_DURATION + 1, dim)
		self.decoder = nn.ModuleList(
			Block(dim, attention_heads, ff_mult, dropout, cross=True)
			for _ in range(config.decoder_layers)
		)
		self.decoder_norm = nn.LayerNorm(dim)

		self.outputs = nn.ModuleList(
			nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, size))
			for size in sizes.values()
		)
		self.chosen = nn.ModuleList(
			nn.Embedding(size + 1, dim) for size in list(sizes.values())[:-1]
		)

	@classmethod
	def _from_pretrained(
		cls,
		*,
		model_id: str,
		revision: str | None,
		cache_dir: str | Path | None,
		force_download: bool,
		local_files_only: bool,
		token: str | bool | None,
		map_location: str = "cpu",
		strict: bool = True,
		**model_kwargs: Any,
	) -> Self:
		"""Load as `PyTorchModelHubMixin` does, but strictly by default.

		The mixin defaults to `strict=False`, which leaves a weight whose name no longer matches
		the code at its random initial value without an error.
		"""
		return super()._from_pretrained(
			model_id=model_id,
			revision=revision,
			cache_dir=cache_dir,
			force_download=force_download,
			local_files_only=local_files_only,
			token=token,
			map_location=map_location,
			strict=strict,
			**model_kwargs,
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

	def event_states(
		self,
		frames: torch.Tensor,
		frame_mask: torch.Tensor,
		starts: torch.Tensor,
		values: torch.Tensor,
		event_mask: torch.Tensor,
	) -> torch.Tensor:
		"""Decoder states (batch, events, dim) of events at `starts`, each seeing the ones before.

		`values` (batch, events, heads) indexes `HEAD_VALUES`, with IGNORE for a head not predicted.
		The state of an event reads only the values of the events before it.
		"""
		size, events, _ = values.shape

		# A head that is not predicted, and every head before the 1st event, gets its own value.
		values = torch.where(values == IGNORE, self.not_predicted, values)
		previous = torch.cat((self.not_predicted.expand(size, 1, -1), values[:, :-1]), dim=1)
		x = frames.gather(1, starts[..., None].expand(-1, -1, frames.shape[-1]))

		for index, embedding in enumerate(self.previous):
			x = x + embedding(previous[..., index])

		lengths = frame_mask.sum(dim=1, keepdim=True)
		x = x + self.remaining((lengths - starts).clamp(max=MAX_DURATION))

		causal = torch.ones(events, events, dtype=torch.bool, device=x.device).tril()
		self_mask = causal & event_mask[:, None]
		cross_mask = frame_mask[:, None].expand(size, events, -1)
		frame_times = torch.arange(frames.shape[1], device=x.device).expand(size, -1)

		for block in self.decoder:
			x = block(x, starts, self_mask, frames, frame_times, cross_mask)

		return self.decoder_norm(x)

	def forward(self, batch: Batch) -> PolicyOutput:
		"""Logits and log π of the target events in `batch`, each event given the ones before."""
		frames = self.encode(batch)
		x = self.event_states(
			frames, batch.frame_mask, batch.starts, batch.values, batch.event_mask
		)
		values = torch.where(batch.values == IGNORE, self.not_predicted, batch.values)

		logits: dict[str, torch.Tensor] = {}
		log_prob = torch.zeros(values.shape[:2], device=x.device)

		for index, (head, output) in enumerate(zip(HEAD_VALUES, self.outputs, strict=True)):
			logits[head] = output(x).float().masked_fill(~batch.masks[head], float("-inf"))
			target = batch.values[..., index]
			chosen = logits[head].log_softmax(-1).gather(-1, target.clamp(min=0)[..., None])
			log_prob = log_prob + torch.where(target == IGNORE, 0.0, chosen.squeeze(-1))

			if index < len(self.chosen):
				x = x + self.chosen[index](values[..., index])

		return PolicyOutput(logits=logits, log_prob=log_prob)

	def harmonize(
		self,
		melody: Melody,
		*,
		temperature: float = 1.0,
		greedy: bool = False,
		generator: torch.Generator | None = None,
	) -> tuple[ChordEvent, ...]:
		"""Chord events for 1 melody, as `generate` chooses them."""
		events = self.generate(
			[melody], temperature=temperature, greedy=greedy, generator=generator
		)
		return events[0]

	@torch.no_grad()
	def generate(
		self,
		melodies: Sequence[Melody],
		*,
		temperature: float = 1.0,
		greedy: bool = False,
		generator: torch.Generator | None = None,
		batch_size: int | None = None,
	) -> list[tuple[ChordEvent, ...]]:
		"""Chord events for each of `melodies`, chosen head by head under the action masks.

		With `greedy`, each head takes its most likely value; otherwise it is sampled
		from its logits divided by `temperature`, drawing from `generator`.
		With `batch_size`, melodies of similar length are generated that many at a time.
		The policy should be run in eval mode.
		"""
		if temperature <= 0:
			raise ValueError(f"temperature must be positive, got {temperature}")

		if batch_size is None:
			return self._generate_batch(melodies, temperature, greedy, generator)

		order = sorted(range(len(melodies)), key=lambda i: melodies[i].length)
		results: list[tuple[ChordEvent, ...]] = [() for _ in melodies]

		for chunk in batched(order, batch_size, strict=False):
			outputs = self._generate_batch(
				[melodies[i] for i in chunk], temperature, greedy, generator
			)

			for i, events in zip(chunk, outputs, strict=True):
				results[i] = events

		return results

	def _generate_batch(
		self,
		melodies: Sequence[Melody],
		temperature: float,
		greedy: bool,
		generator: torch.Generator | None,
	) -> list[tuple[ChordEvent, ...]]:
		device = self.not_predicted.device
		heads = tuple(HEAD_VALUES)
		batch = collate([(melody_features(melody), _no_events()) for melody in melodies]).to(device)
		frames = self.encode(batch)

		size = len(melodies)
		lengths = [melody.length for melody in melodies]
		positions = [0] * size

		events: list[list[ChordEvent]] = [[] for _ in melodies]
		starts = torch.zeros(size, 0, dtype=torch.long, device=device)
		values = torch.zeros(size, 0, len(heads), dtype=torch.long, device=device)

		while active := [b for b in range(size) if positions[b] < lengths[b]]:
			step = [min(p, n - 1) for p, n in zip(positions, lengths, strict=True)]
			starts = torch.cat((starts, torch.tensor(step, device=device)[:, None]), dim=1)
			blank = torch.full((size, 1, len(heads)), IGNORE, dtype=torch.long, device=device)
			values = torch.cat((values, blank), dim=1)
			counts = torch.tensor(
				[len(events[b]) + (b in active) for b in range(size)], device=device
			)
			event_mask = torch.arange(values.shape[1], device=device) < counts[:, None]
			x = self.event_states(frames, batch.frame_mask, starts, values, event_mask)[:, -1]

			chosen: list[dict[str, Any]] = [{} for _ in range(size)]
			for index, head in enumerate(heads):
				predicted = [
					b
					for b in active
					if head in ("root", "duration") or chosen[b]["root"] is not None
				]

				if predicted:
					masks = [
						allowed(
							head,
							chosen[b],
							previous_no_chord=bool(events[b]) and events[b][-1].chord is None,
							remaining=lengths[b] - positions[b],
						)
						for b in predicted
					]
					allowed_values = torch.tensor(masks, dtype=torch.bool, device=device)
					logits = self.outputs[index](x[predicted]).float()
					logits = logits.masked_fill(~allowed_values, float("-inf"))
					picks = _pick(logits, temperature, greedy, generator)

					for b, value in zip(predicted, picks, strict=True):
						chosen[b][head] = HEAD_VALUES[head][value]
						values[b, -1, index] = value

				if index < len(self.chosen):
					column = values[:, -1, index]
					x = x + self.chosen[index](
						torch.where(column == IGNORE, self.not_predicted[index], column)
					)

			for b in active:
				prefix = chosen[b]
				chord = None
				if prefix["root"] is not None:
					chord = RawChord(**{head: prefix[head] for head in heads if head != "duration"})

				events[b].append(ChordEvent(chord, prefix["duration"]))
				positions[b] += prefix["duration"]

		return [tuple(e) for e in events]


def _no_events() -> EventTargets:
	return EventTargets(
		starts=np.zeros(0, dtype=np.int64),
		values=np.full((0, len(HEAD_VALUES)), IGNORE, dtype=np.int64),
		masks={head: np.ones((0, len(v)), dtype=bool) for head, v in HEAD_VALUES.items()},
	)


def _pick(
	logits: torch.Tensor, temperature: float, greedy: bool, generator: torch.Generator | None
) -> list[int]:
	if greedy:
		return logits.argmax(-1).tolist()

	probabilities = (logits / temperature).softmax(-1)
	return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1).tolist()
