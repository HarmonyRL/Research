"""Train the structure critic on human accompaniments against policy samples."""

import hashlib
import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.data import read_segments
from harmonyrl.models import ChordCritic, CriticArguments, CriticConfig, CriticTrainer
from harmonyrl.schema import ChordEvent, Segment, Split, no_chord_limit

REPO = Path(__file__).resolve().parent.parent


@hydra.main(config_path="../configs", config_name="train_critic")
def main(cfg: DictConfig) -> None:
	logging.basicConfig(level=logging.INFO, format="%(message)s")

	config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
	started = datetime.now(ZoneInfo("Asia/Seoul"))
	run_dir = Path(HydraConfig.get().runtime.output_dir)
	if any(run_dir.iterdir()):
		raise FileExistsError(f"run directory {run_dir} is not empty")

	meta: dict[str, Any] = {
		"git": _git("rev-parse", "--short", "HEAD"),
		"git_dirty": bool(_git("status", "--porcelain")),
		"started": started.isoformat(timespec="seconds"),
		"config": config,
	}
	_write_json(run_dir / "meta.json", meta)

	data_dir = Path(config["data_dir"])
	splits = (Split.TRAIN, Split.VALID)
	segments = {s: read_segments(data_dir / f"segments_{s.value}.jsonl") for s in splits}

	if config["restrict_no_chord"]:
		segments = {s: [x for x in v if _keeps_no_chord_to_pickup(x)] for s, v in segments.items()}

	samples: defaultdict[str, list[tuple[ChordEvent, ...]]] = defaultdict(list)

	for samples_dir in map(Path, config["samples_dirs"]):
		for s in splits:
			for segment in read_segments(samples_dir / f"segments_{s.value}.jsonl"):
				samples[segment.id].append(segment.events)

	train, valid = segments[Split.TRAIN], segments[Split.VALID]

	if config["split"] == "holdout":
		train, valid = _holdout(train + valid, config["holdout_fraction"], config["seed"])

	torch.manual_seed(config["seed"])
	critic = ChordCritic(CriticConfig(**config["model"]))
	trainer = CriticTrainer(
		critic,
		CriticArguments(output_dir=run_dir, seed=config["seed"], **config["train"]),
		train=train,
		valid=valid,
		train_samples=[samples[s.id] for s in train],
		valid_samples=[samples[s.id] for s in valid],
	)

	logging.info(
		"%d parameters, %d train and %d valid segments",
		sum(p.numel() for p in critic.parameters()),
		len(train),
		len(valid),
	)

	best = trainer.train()
	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	meta["best"] = best
	_write_json(run_dir / "meta.json", meta)

	logging.info(
		"best epoch %d, valid loss %.4f, in %s", best["epoch"], best["valid_loss"], run_dir / "best"
	)


def _keeps_no_chord_to_pickup(segment: Segment) -> bool:
	"""Whether no chord sounds only as the 1st event, within the `no_chord_limit` of the melody."""
	first = segment.events[0]
	return all(e.chord is not None for e in segment.events[1:]) and (
		first.chord is not None or first.duration <= no_chord_limit(segment.melody)
	)


def _holdout(
	segments: list[Segment], fraction: float, seed: int
) -> tuple[list[Segment], list[Segment]]:
	"""Segments whose salted id hash falls below `fraction` validate; the rest train."""

	def share(segment: Segment) -> float:
		digest = hashlib.sha256(f"{seed}:{segment.id}".encode()).digest()
		return int.from_bytes(digest[:8]) / 2**64

	train = [s for s in segments if share(s) >= fraction]
	valid = [s for s in segments if share(s) < fraction]

	return train, valid


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
