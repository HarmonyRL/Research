"""Batches of reward inputs as padded tensors."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class RewardBatch:
	"""Padded token ids of several harmonizations, (batch, tokens), the mask 1 on real tokens."""

	input_ids: torch.Tensor
	attention_mask: torch.Tensor

	def to(self, device: torch.device | str) -> RewardBatch:
		"""The same batch on `device`."""
		return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def collate(sequences: Sequence[np.ndarray]) -> RewardBatch:
	"""Pad the token ids of several harmonizations into 1 batch."""
	ids = [torch.from_numpy(s) for s in sequences]

	return RewardBatch(
		input_ids=pad_sequence(ids, batch_first=True),
		attention_mask=pad_sequence([torch.ones_like(i) for i in ids], batch_first=True),
	)
