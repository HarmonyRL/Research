"""Batches of segments as padded tensors, grouped by length."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from harmonyrl.schema import HEAD_VALUES

from .features import IGNORE, EventTargets, MelodyFeatures


@dataclass(frozen=True)
class Batch:
	"""Padded model inputs and targets for several segments."""

	pitch_classes: torch.Tensor
	onset: torch.Tensor
	octave: torch.Tensor
	bar_position: torch.Tensor
	beat_strength: torch.Tensor
	mode: torch.Tensor
	key_start: torch.Tensor
	frame_mask: torch.Tensor
	beats_per_bar: torch.Tensor
	beat_unit: torch.Tensor
	section: torch.Tensor
	tempo: torch.Tensor
	starts: torch.Tensor
	values: torch.Tensor
	masks: dict[str, torch.Tensor]
	event_mask: torch.Tensor

	def to(self, device: torch.device | str) -> Batch:
		"""The same batch on `device`."""
		moved: dict[str, object] = {}
		for field in fields(self):
			value = getattr(self, field.name)
			if isinstance(value, dict):
				moved[field.name] = {head: mask.to(device) for head, mask in value.items()}
			else:
				moved[field.name] = value.to(device)

		return replace(self, **moved)


def _pad(arrays: Sequence[np.ndarray], value: float = 0) -> torch.Tensor:
	return pad_sequence(
		[torch.from_numpy(a) for a in arrays], batch_first=True, padding_value=value
	)


def collate(items: Sequence[tuple[MelodyFeatures, EventTargets]]) -> Batch:
	"""Pad the features and targets of several segments into 1 batch."""
	features = [f for f, _ in items]
	targets = [t for _, t in items]

	return Batch(
		pitch_classes=_pad([f.pitch_classes for f in features]),
		onset=_pad([f.onset for f in features]),
		octave=_pad([f.octave for f in features]),
		bar_position=_pad([f.bar_position for f in features]),
		beat_strength=_pad([f.beat_strength for f in features]),
		mode=_pad([f.mode for f in features]),
		key_start=_pad([f.key_start for f in features]),
		frame_mask=_pad([np.ones(len(f.onset), dtype=bool) for f in features]),
		beats_per_bar=torch.tensor([f.beats_per_bar for f in features]),
		beat_unit=torch.tensor([f.beat_unit for f in features]),
		section=torch.tensor([f.section for f in features]),
		tempo=torch.tensor([f.tempo for f in features]),
		starts=_pad([t.starts for t in targets]),
		values=_pad([t.values for t in targets], IGNORE),
		masks={head: _pad([t.masks[head] for t in targets], True) for head in HEAD_VALUES},
		event_mask=_pad([np.ones(len(t.starts), dtype=bool) for t in targets]),
	)


def length_batches(
	lengths: Sequence[int], max_frames: int, rng: np.random.Generator
) -> list[list[int]]:
	"""Indices grouped so that each batch pads to at most `max_frames` frames, in random order."""
	batches: list[list[int]] = []
	current: list[int] = []
	order = np.lexsort((rng.random(len(lengths)), np.asarray(lengths)))

	for index in order.tolist():
		if current and lengths[index] * (len(current) + 1) > max_frames:
			batches.append(current)
			current = []
		current.append(index)

	if current:
		batches.append(current)

	rng.shuffle(batches)
	return batches
