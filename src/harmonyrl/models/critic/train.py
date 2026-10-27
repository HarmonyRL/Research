"""Training of the structure critic on human accompaniments against policy samples."""

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from harmonyrl.schema import ChordEvent, Segment

from ..batch import length_batches
from ..features import MelodyFeatures, melody_features
from ..schedule import warmup_cosine
from .batch import collate
from .features import ChordFrames, chord_frames
from .model import ChordCritic

_logger = logging.getLogger(__name__)

type Example = tuple[MelodyFeatures, ChordFrames]


@dataclass(frozen=True)
class CriticArguments:
	"""Settings of `CriticTrainer`, as `transformers.TrainingArguments` for the options used here.

	Batches hold at most `max_frames` padded frames. The learning rate rises linearly over
	`warmup_steps`, then follows a cosine to 0 at the end of `max_epochs`. Training stops once
	the validation loss has not improved for `patience` epochs. Targets move `label_smoothing` / 2
	toward the other class. `section_dropout` is the share of training examples whose section is
	replaced by unknown, as melodies from outside the dataset have none.
	"""

	output_dir: str | Path
	seed: int = 0
	max_frames: int = 8192
	lr: float = 3.0e-4
	weight_decay: float = 0.1
	warmup_steps: int = 300
	max_epochs: int = 20
	patience: int = 4
	grad_clip: float = 1.0
	label_smoothing: float = 0.1
	section_dropout: float = 0.2


class CriticTrainer:
	"""Training of a `ChordCritic`, shaped like `transformers.Trainer`.

	`train_samples` and `valid_samples` hold the policy samples of each segment, the negatives.
	Each epoch pairs every training segment with 1 of its samples drawn afresh; the validation
	samples are drawn once. A sample that gives the critic the same input as the human
	accompaniment is left out, as it would be a positive labeled negative, and a segment left
	without samples has no negative. `train` writes the metrics of each epoch to `metrics.jsonl`
	and the critic with the lowest validation loss to `best/` in `args.output_dir`, and ends with
	those weights loaded.
	"""

	def __init__(
		self,
		critic: ChordCritic,
		args: CriticArguments,
		train: Sequence[Segment],
		valid: Sequence[Segment],
		train_samples: Sequence[Sequence[Sequence[ChordEvent]]],
		valid_samples: Sequence[Sequence[Sequence[ChordEvent]]],
		device: str | None = None,
	) -> None:
		self.critic = critic
		self.args = args
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self._rng = np.random.default_rng(args.seed)

		self._train = list(train)
		self._train_melodies = [melody_features(s.melody) for s in train]
		self._train_humans = [chord_frames(s.melody, s.events) for s in train]
		self._train_samples = [
			(index, distinct)
			for index, (segment, human, samples) in enumerate(
				zip(self._train, self._train_humans, train_samples, strict=True)
			)
			if (distinct := _distinct(segment, human, samples))
		]

		examples = [(melody_features(s.melody), chord_frames(s.melody, s.events)) for s in valid]
		labels = [1.0] * len(valid)

		for (melody, human), segment, samples in zip(
			list(examples), valid, valid_samples, strict=True
		):
			if distinct := _distinct(segment, human, samples):
				events = distinct[self._rng.integers(len(distinct))]
				examples.append((melody, chord_frames(segment.melody, events)))
				labels.append(0.0)

		self._valid_labels = np.array(labels)
		self._valid_batches = [
			(collate([examples[i] for i in indices], [labels[i] for i in indices]), indices)
			for indices in length_batches(
				[len(m.onset) for m, _ in examples], args.max_frames, self._rng
			)
		]

	def train(self) -> dict[str, float]:
		"""Train until early stopping or `max_epochs`. Returns the best epoch and its loss."""
		args = self.args
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		torch.manual_seed(args.seed)

		critic = self.critic.to(self.device)

		lengths = [len(m.onset) for m in self._train_melodies]
		lengths += [lengths[index] for index, _ in self._train_samples]
		labels = [1.0] * len(self._train) + [0.0] * len(self._train_samples)

		optimizer = torch.optim.AdamW(
			critic.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
			fused=self.device.type == "cuda",
		)
		steps_per_epoch = len(length_batches(lengths, args.max_frames, self._rng))
		scheduler = torch.optim.lr_scheduler.LambdaLR(
			optimizer, warmup_cosine(args.warmup_steps, steps_per_epoch * args.max_epochs)
		)

		best_loss = math.inf
		best_epoch = 0
		best_state: dict[str, torch.Tensor] = {}
		metrics_file = output_dir / "metrics.jsonl"
		metrics_file.write_text("", encoding="utf-8")

		for epoch in range(1, args.max_epochs + 1):
			epoch_start = time.perf_counter()
			examples = self._epoch_examples()

			critic.train()

			loss_sum = 0.0

			for indices in length_batches(lengths, args.max_frames, self._rng):
				batch = collate([examples[i] for i in indices], [labels[i] for i in indices])
				batch = batch.to(self.device)
				dropped = torch.rand(batch.section.shape, device=self.device) < args.section_dropout
				batch = replace(batch, section=batch.section.masked_fill(dropped, 0))
				targets = batch.labels * (1 - args.label_smoothing) + args.label_smoothing / 2

				with torch.autocast(self.device.type, dtype=torch.bfloat16):
					logits = critic(batch)

				loss = functional.binary_cross_entropy_with_logits(logits.float(), targets)
				optimizer.zero_grad(set_to_none=True)
				loss.backward()

				torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)

				optimizer.step()
				scheduler.step()

				loss_sum += loss.item() * len(indices)

			valid = self.evaluate()
			metrics = {
				"epoch": epoch,
				"lr": scheduler.get_last_lr()[0],
				"train_loss": loss_sum / len(lengths),
				**{f"valid_{k}": v for k, v in valid.items()},
				"seconds": round(time.perf_counter() - epoch_start, 1),
			}

			with metrics_file.open("a", encoding="utf-8") as f:
				f.write(json.dumps(metrics) + "\n")

			_logger.info(
				"epoch %d train %.4f valid %.4f accuracy %.3f auc %.3f ece %.3f (%.1fs)",
				epoch,
				metrics["train_loss"],
				valid["loss"],
				valid["accuracy"],
				valid["auc"],
				valid["ece"],
				metrics["seconds"],
			)

			if valid["loss"] < best_loss:
				best_loss = valid["loss"]
				best_epoch = epoch
				best_state = {k: v.detach().clone() for k, v in critic.state_dict().items()}
				critic.save_pretrained(output_dir / "best")
			elif epoch - best_epoch >= args.patience:
				break

		critic.load_state_dict(best_state)
		return {"epoch": best_epoch, "valid_loss": best_loss}

	@torch.no_grad()
	def evaluate(self) -> dict[str, float]:
		"""Validation loss without smoothing, accuracy, AUC, and calibration error.

		The AUC is the chance that a human accompaniment scores above a sample, ties counting half.
		The calibration error is the expected calibration error over 10 equal-width bins of the
		predicted probability of a human accompaniment.
		"""
		critic = self.critic.to(self.device)
		critic.eval()

		labels = self._valid_labels
		probabilities = np.zeros(len(labels))
		loss_sum = 0.0

		for cpu_batch, indices in self._valid_batches:
			batch = cpu_batch.to(self.device)

			with torch.autocast(self.device.type, dtype=torch.bfloat16):
				logits = critic(batch).float()

			loss_sum += functional.binary_cross_entropy_with_logits(
				logits, batch.labels, reduction="sum"
			).item()

			probabilities[indices] = logits.sigmoid().cpu().numpy()

		human = labels > 0.5
		_, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
		ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
		humans, samples = human.sum(), (~human).sum()
		bins = np.minimum((probabilities * 10).astype(int), 9)
		ece = sum(
			abs(probabilities[bins == b].mean() - labels[bins == b].mean()) * (bins == b).mean()
			for b in range(10)
			if (bins == b).any()
		)

		return {
			"loss": loss_sum / len(labels),
			"accuracy": float(((probabilities > 0.5) == human).mean()),
			"auc": float((ranks[human].sum() - humans * (humans + 1) / 2) / (humans * samples)),
			"ece": float(ece),
		}

	def _epoch_examples(self) -> list[Example]:
		"""The training humans followed by 1 fresh sample of each segment that has one."""
		examples: list[Example] = list(zip(self._train_melodies, self._train_humans, strict=True))

		for index, samples in self._train_samples:
			events = samples[self._rng.integers(len(samples))]
			frames = chord_frames(self._train[index].melody, events)
			examples.append((self._train_melodies[index], frames))

		return examples


def _distinct(
	segment: Segment, human: ChordFrames, samples: Sequence[Sequence[ChordEvent]]
) -> list[Sequence[ChordEvent]]:
	"""The samples of `segment` whose frames differ from `human`, the human accompaniment's."""
	distinct = []

	for events in samples:
		frames = chord_frames(segment.melody, events)
		if not all(
			np.array_equal(getattr(frames, f.name), getattr(human, f.name))
			for f in fields(ChordFrames)
		):
			distinct.append(events)

	return distinct
