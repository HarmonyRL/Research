"""Batches of critic inputs as padded tensors."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ..features import MelodyFeatures
from .features import ChordFrames


@dataclass(frozen=True)
class CriticBatch:
	"""Padded inputs for several harmonizations and their labels, 1 for a human accompaniment.

	Frame tensors are (batch, frames), with `pitch_classes` and `chord_tones` adding 12 pitch
	classes and `chord_values` the chord heads. `frame_mask` is True on real frames.
	"""

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
	chord_values: torch.Tensor
	chord_tones: torch.Tensor
	chord_bass: torch.Tensor
	chord_onset: torch.Tensor
	labels: torch.Tensor

	def to(self, device: torch.device | str) -> CriticBatch:
		"""The same batch on `device`."""
		return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def collate(
	items: Sequence[tuple[MelodyFeatures, ChordFrames]], labels: Sequence[float]
) -> CriticBatch:
	"""Pad the inputs of several harmonizations into 1 batch."""
	melodies = [m for m, _ in items]
	chords = [c for _, c in items]

	return CriticBatch(
		pitch_classes=_pad([m.pitch_classes for m in melodies]),
		onset=_pad([m.onset for m in melodies]),
		octave=_pad([m.octave for m in melodies]),
		bar_position=_pad([m.bar_position for m in melodies]),
		beat_strength=_pad([m.beat_strength for m in melodies]),
		mode=_pad([m.mode for m in melodies]),
		key_start=_pad([m.key_start for m in melodies]),
		frame_mask=_pad([np.ones(len(m.onset), dtype=bool) for m in melodies]),
		beats_per_bar=torch.tensor([m.beats_per_bar for m in melodies]),
		beat_unit=torch.tensor([m.beat_unit for m in melodies]),
		section=torch.tensor([m.section for m in melodies]),
		chord_values=_pad([c.values for c in chords]),
		chord_tones=_pad([c.tones for c in chords]),
		chord_bass=_pad([c.bass for c in chords]),
		chord_onset=_pad([c.onset for c in chords]),
		labels=torch.tensor(labels, dtype=torch.float32),
	)


def _pad(arrays: Sequence[np.ndarray]) -> torch.Tensor:
	return pad_sequence([torch.from_numpy(a) for a in arrays], batch_first=True)
