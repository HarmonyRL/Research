"""Frozen features of a harmonization: the log probabilities or decoder states of the chord policy,
or the states of a pretrained symbolic-music encoder."""

from collections.abc import Iterator, Sequence
from itertools import batched

import torch
from miditok import MusicTokenizer
from torch import nn
from transformers import BertConfig, BertModel

from harmonyrl.schema import ChordEvent, Melody

from ..batch import Batch
from ..batch import collate as policy_collate
from ..features import IGNORE, melody_features, prefix_targets
from ..policy import ChordPolicy
from .batch import collate
from .tokens import token_ids


def _policy_batches(
	policy: ChordPolicy,
	melodies: Sequence[Melody],
	harmonizations: Sequence[Sequence[ChordEvent]],
	batch_size: int,
) -> Iterator[tuple[tuple[int, ...], Batch]]:
	"""Batches of `harmonizations` among similar lengths, each with its indices."""
	device = next(policy.parameters()).device
	restrict = policy.config.restrict_no_chord
	items = [
		(melody_features(m), prefix_targets(m, e, restrict_no_chord=restrict))
		for m, e in zip(melodies, harmonizations, strict=True)
	]
	order = sorted(range(len(items)), key=lambda i: melodies[i].length, reverse=True)

	for chunk in batched(order, batch_size, strict=False):
		yield chunk, policy_collate([items[i] for i in chunk]).to(device)


class LikelihoodFeatures(nn.Module):
	"""Mean and lowest log probability of the chord events of a harmonization under the frozen
	`policy`."""

	dim = 2

	def __init__(self, policy: ChordPolicy) -> None:
		super().__init__()
		self.policy = policy.requires_grad_(False)

	@torch.no_grad()
	def forward(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int,
	) -> torch.Tensor:
		"""Features (harmonizations, 2), `batch_size` at a time among similar lengths."""
		policy = self.policy.eval()
		device = next(policy.parameters()).device
		features = torch.zeros(len(melodies), self.dim, device=device)

		for chunk, batch in _policy_batches(policy, melodies, harmonizations, batch_size):
			log_prob = policy(batch).log_prob
			mask = batch.event_mask
			mean = (log_prob * mask).sum(dim=1) / mask.sum(dim=1)
			lowest = log_prob.masked_fill(~mask, torch.inf).min(dim=1).values
			features[list(chunk)] = torch.stack((mean, lowest), dim=1).float()

		return features


class PolicyFeatures(nn.Module):
	"""Mean decoder state of the frozen `policy` after each chord event of a harmonization."""

	def __init__(self, policy: ChordPolicy) -> None:
		super().__init__()
		self.policy = policy.requires_grad_(False)
		self.dim = policy.config.dim

	@torch.no_grad()
	def forward(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int,
	) -> torch.Tensor:
		"""Features (harmonizations, dim), `batch_size` at a time among similar lengths."""
		policy = self.policy.eval()
		device = next(policy.parameters()).device
		features = torch.zeros(len(melodies), self.dim, device=device)

		for chunk, batch in _policy_batches(policy, melodies, harmonizations, batch_size):
			size = len(chunk)
			rows = torch.arange(size, device=device)
			counts = batch.event_mask.sum(dim=1)

			starts = torch.cat((batch.starts, batch.starts.new_zeros(size, 1)), dim=1)
			starts[rows, counts] = batch.frame_mask.sum(dim=1) - 1
			end = batch.values.new_full((size, 1, batch.values.shape[2]), IGNORE)

			values = torch.cat((batch.values, end), dim=1)
			mask = torch.cat((batch.event_mask, batch.event_mask.new_zeros(size, 1)), dim=1)
			mask[rows, counts] = True

			states = policy.event_states(
				policy.encode(batch), batch.frame_mask, starts, values, mask
			)
			after = mask.clone()
			after[:, 0] = False
			weights = after[..., None].to(states.dtype)
			features[list(chunk)] = ((states * weights).sum(dim=1) / weights.sum(dim=1)).float()

		return features


class MusicbertFeatures(nn.Module):
	"""Mean-pooled states of a frozen encoder over the REMI+BPE tokens of the harmonization
	played with the fixed voicing."""

	def __init__(self, backbone: str, tokenizer: str, max_tokens: int, pretrained: bool) -> None:
		super().__init__()
		config = BertConfig.from_pretrained(backbone)
		encoder = (
			BertModel.from_pretrained(backbone, add_pooling_layer=False)
			if pretrained
			else BertModel(config, add_pooling_layer=False)
		)

		self.tokenizer = MusicTokenizer.from_pretrained(tokenizer)
		self.max_tokens = max_tokens
		self.encoder = encoder.requires_grad_(False)
		self.dim = config.hidden_size

	@torch.no_grad()
	def forward(
		self,
		melodies: Sequence[Melody],
		harmonizations: Sequence[Sequence[ChordEvent]],
		batch_size: int,
	) -> torch.Tensor:
		"""Features (harmonizations, dim), `batch_size` at a time among similar lengths."""
		encoder = self.encoder.eval()
		device = next(encoder.parameters()).device
		inputs = [
			token_ids(self.tokenizer, m, e, self.max_tokens)
			for m, e in zip(melodies, harmonizations, strict=True)
		]
		order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]), reverse=True)
		features = torch.zeros(len(inputs), self.dim, device=device)

		for chunk in batched(order, batch_size, strict=False):
			batch = collate([inputs[i] for i in chunk]).to(device)
			states = encoder(
				input_ids=batch.input_ids, attention_mask=batch.attention_mask
			).last_hidden_state
			weights = batch.attention_mask[..., None].to(states.dtype)
			features[list(chunk)] = ((states * weights).sum(dim=1) / weights.sum(dim=1)).float()

		return features
