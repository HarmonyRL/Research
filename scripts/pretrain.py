"""Pretrain the chord policy π_ref on human accompaniments with teacher forcing.

Writes the config, git state, per-epoch metrics, and the checkpoint with the lowest validation NLL
per event into a new run directory. Training stops early when that NLL stops improving.
"""

import argparse
import json
import math
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from harmonyrl.data import read_segments
from harmonyrl.models import (
	IGNORE,
	Batch,
	ChordPolicy,
	EventTargets,
	MelodyFeatures,
	collate,
	event_targets,
	head_values_json,
	length_batches,
	melody_features,
)
from harmonyrl.schema import HEAD_VALUES, Split

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/pretrain.yaml", type=Path)
	parser.add_argument("--name", help="run directory name, the start time by default")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	train_config = config["train"]
	started = datetime.now(UTC)

	run_dir = Path(config["run_dir"]) / (args.name or started.strftime("%Y%m%d-%H%M%S"))
	run_dir.mkdir(parents=True)

	meta: dict[str, Any] = {
		"git": _git("rev-parse", "--short", "HEAD"),
		"git_dirty": bool(_git("status", "--porcelain")),
		"started": started.isoformat(timespec="seconds"),
		"config": config,
	}
	_write_json(run_dir / "meta.json", meta)

	torch.manual_seed(config["seed"])
	rng = np.random.default_rng(config["seed"])
	device = "cuda" if torch.cuda.is_available() else "cpu"

	data_dir = Path(config["data_dir"])
	train = _load(data_dir / f"segments_{Split.TRAIN.value}.jsonl")
	valid = _load(data_dir / f"segments_{Split.VALID.value}.jsonl")
	train_lengths = [len(features.onset) for features, _ in train]
	valid_lengths = [len(features.onset) for features, _ in valid]
	valid_batches = [
		collate([valid[i] for i in indices])
		for indices in length_batches(valid_lengths, train_config["max_frames"], rng)
	]

	model = ChordPolicy(head_values_json(), **config["model"]).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=train_config["lr"],
		weight_decay=train_config["weight_decay"],
		fused=True,
	)
	steps_per_epoch = len(length_batches(train_lengths, train_config["max_frames"], rng))
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		_warmup_cosine(train_config["warmup_steps"], steps_per_epoch * train_config["max_epochs"]),
	)
	print(f"{sum(p.numel() for p in model.parameters())} parameters, {steps_per_epoch} steps/epoch")

	best_nll = math.inf
	best_epoch = 0
	for epoch in range(1, train_config["max_epochs"] + 1):
		epoch_start = time.perf_counter()
		model.train()
		nll_sum = 0.0
		event_count = 0
		for indices in length_batches(train_lengths, train_config["max_frames"], rng):
			batch = collate([train[i] for i in indices]).to(device)
			dropped = (
				torch.rand(batch.section.shape, device=device) < train_config["section_dropout"]
			)
			batch = replace(batch, section=batch.section.masked_fill(dropped, 0))
			with torch.autocast(device, dtype=torch.bfloat16):
				log_prob = model(batch).log_prob[batch.event_mask]
			loss = -log_prob.mean()
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
			optimizer.step()
			scheduler.step()
			nll_sum -= log_prob.sum().item()
			event_count += log_prob.numel()

		metrics = {
			"epoch": epoch,
			"lr": scheduler.get_last_lr()[0],
			"train_nll": nll_sum / event_count,
			**{f"valid_{k}": v for k, v in _evaluate(model, valid_batches, device, False).items()},
			**{
				f"valid_no_section_{k}": v
				for k, v in _evaluate(model, valid_batches, device, True).items()
			},
			"seconds": round(time.perf_counter() - epoch_start, 1),
		}
		with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
			f.write(json.dumps(metrics) + "\n")
		print(
			f"epoch {epoch} train {metrics['train_nll']:.4f} valid {metrics['valid_nll']:.4f} "
			f"no-section {metrics['valid_no_section_nll']:.4f} ({metrics['seconds']}s)"
		)

		if metrics["valid_nll"] < best_nll:
			best_nll = metrics["valid_nll"]
			best_epoch = epoch
			model.save_pretrained(run_dir / "best")
		elif epoch - best_epoch >= train_config["patience"]:
			break

	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	meta["best"] = {"epoch": best_epoch, "valid_nll": best_nll}
	_write_json(run_dir / "meta.json", meta)
	print(f"best epoch {best_epoch}, valid NLL {best_nll:.4f}, saved to {run_dir / 'best'}")


def _load(path: Path) -> list[tuple[MelodyFeatures, EventTargets]]:
	return [(melody_features(s.melody), event_targets(s)) for s in read_segments(path)]


@torch.no_grad()
def _evaluate(
	model: ChordPolicy, batches: Sequence[Batch], device: str, unknown_section: bool
) -> dict[str, float]:
	"""NLL per event and teacher-forced accuracy of each head over `batches`."""
	model.eval()
	nll_sum = 0.0
	event_count = 0
	correct = dict.fromkeys(HEAD_VALUES, 0)
	counts = dict.fromkeys(HEAD_VALUES, 0)
	for cpu_batch in batches:
		batch = cpu_batch.to(device)
		if unknown_section:
			batch = replace(batch, section=torch.zeros_like(batch.section))
		with torch.autocast(device, dtype=torch.bfloat16):
			output = model(batch)
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


def _warmup_cosine(warmup: int, total: int) -> Callable[[int], float]:
	"""Learning rate factor: linear warmup, then cosine decay to 0 at step `total`."""

	def factor(step: int) -> float:
		if step < warmup:
			return (step + 1) / warmup
		progress = min(1.0, (step - warmup) / max(1, total - warmup))
		return 0.5 * (1 + math.cos(math.pi * progress))

	return factor


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
