"""Pretraining of the chord policy on human accompaniments with teacher forcing."""

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from harmonyrl.schema import HEAD_VALUES, Segment

from ..batch import Batch, collate, length_batches
from ..features import IGNORE, event_targets, melody_features
from ..schedule import warmup_cosine
from .model import ChordPolicy

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainArguments:
	"""Settings of `PolicyTrainer`, as `transformers.TrainingArguments` for the options used here."""

	output_dir: str | Path
	seed: int = 0
	max_frames: int = 8192
	lr: float = 3.0e-4
	weight_decay: float = 0.1
	warmup_steps: int = 500
	max_epochs: int = 15
	patience: int = 5
	grad_clip: float = 1.0
	section_dropout: float = 0.2
	tempo_dropout: float = 0.2


class PolicyTrainer:
	"""Teacher-forced training of a `ChordPolicy`, shaped like `transformers.Trainer`."""

	def __init__(
		self,
		policy: ChordPolicy,
		args: PretrainArguments,
		train: Sequence[Segment],
		valid: Sequence[Segment],
		device: str | None = None,
	) -> None:
		self.policy = policy
		self.args = args
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self._rng = np.random.default_rng(args.seed)

		self._train = [(melody_features(s.melody), event_targets(s)) for s in train]
		valid_items = [(melody_features(s.melody), event_targets(s)) for s in valid]
		self._valid_batches = [
			collate([valid_items[i] for i in indices])
			for indices in length_batches(
				[len(f.onset) for f, _ in valid_items], args.max_frames, self._rng
			)
		]

	def train(self) -> dict[str, float]:
		"""Train until early stopping or `max_epochs`. Returns the best epoch and its NLL."""
		args = self.args
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		torch.manual_seed(args.seed)

		policy = self.policy.to(self.device)
		lengths = [len(features.onset) for features, _ in self._train]
		optimizer = torch.optim.AdamW(
			policy.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
			fused=self.device.type == "cuda",
		)
		steps_per_epoch = len(length_batches(lengths, args.max_frames, self._rng))
		scheduler = torch.optim.lr_scheduler.LambdaLR(
			optimizer, warmup_cosine(args.warmup_steps, steps_per_epoch * args.max_epochs)
		)

		best_nll = math.inf
		best_epoch = 0
		best_state: dict[str, torch.Tensor] = {}

		metrics_file = output_dir / "metrics.jsonl"
		metrics_file.write_text("", encoding="utf-8")

		for epoch in range(1, args.max_epochs + 1):
			epoch_start = time.perf_counter()

			policy.train()

			nll_sum = 0.0
			event_count = 0

			for indices in length_batches(lengths, args.max_frames, self._rng):
				batch = collate([self._train[i] for i in indices]).to(self.device)
				drops = torch.rand(2, len(indices), device=self.device)
				batch = replace(
					batch,
					section=batch.section.masked_fill(drops[0] < args.section_dropout, 0),
					tempo=batch.tempo.masked_fill(drops[1] < args.tempo_dropout, 0),
				)

				with torch.autocast(self.device.type, dtype=torch.bfloat16):
					log_prob = policy(batch).log_prob[batch.event_mask]

				loss = -log_prob.mean()
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

				optimizer.step()
				scheduler.step()

				nll_sum -= log_prob.sum().item()
				event_count += log_prob.numel()

			valid = self.evaluate()
			no_section = self.evaluate(unknown_section=True)
			metrics = {
				"epoch": epoch,
				"lr": scheduler.get_last_lr()[0],
				"train_nll": nll_sum / event_count,
				**{f"valid_{k}": v for k, v in valid.items()},
				**{f"valid_no_section_{k}": v for k, v in no_section.items()},
				"seconds": round(time.perf_counter() - epoch_start, 1),
			}

			with metrics_file.open("a", encoding="utf-8") as f:
				f.write(json.dumps(metrics) + "\n")

			_logger.info(
				"epoch %d train %.4f valid %.4f no-section %.4f (%.1fs)",
				epoch,
				nll_sum / event_count,
				valid["nll"],
				no_section["nll"],
				metrics["seconds"],
			)

			if valid["nll"] < best_nll:
				best_nll = valid["nll"]
				best_epoch = epoch
				best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

				policy.save_pretrained(output_dir / "best")
			elif epoch - best_epoch >= args.patience:
				break

		policy.load_state_dict(best_state)
		return {"epoch": best_epoch, "valid_nll": best_nll}

	@torch.no_grad()
	def evaluate(self, unknown_section: bool = False) -> dict[str, float]:
		"""Validation NLL per event and teacher-forced accuracy of each head.

		With `unknown_section`, every segment is evaluated as if its section were unknown.
		"""
		policy = self.policy.to(self.device)
		policy.eval()

		nll_sum = 0.0
		event_count = 0
		correct = dict.fromkeys(HEAD_VALUES, 0)
		counts = dict.fromkeys(HEAD_VALUES, 0)

		for cpu_batch in self._valid_batches:
			batch: Batch = cpu_batch.to(self.device)

			if unknown_section:
				batch = replace(batch, section=torch.zeros_like(batch.section))

			with torch.autocast(self.device.type, dtype=torch.bfloat16):
				output = policy(batch)

			nll_sum -= output.log_prob[batch.event_mask].sum().item()
			event_count += int(batch.event_mask.sum())

			for index, head in enumerate(HEAD_VALUES):
				target = batch.values[..., index]
				predicted = target != IGNORE
				hits = output.logits[head].argmax(-1) == target

				correct[head] += int(hits[predicted].sum())
				counts[head] += int(predicted.sum())

		return {"nll": nll_sum / event_count} | {
			f"acc_{head}": correct[head] / counts[head] for head in HEAD_VALUES
		}
