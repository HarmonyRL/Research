"""Pretrain the chord policy π_ref on human accompaniments with teacher forcing.

Writes the config, git state, per-epoch metrics, and the checkpoint with the lowest validation NLL
per event into a new run directory. Training stops early when that NLL stops improving.
"""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from harmonyrl.data import read_segments
from harmonyrl.models import ChordPolicy, PolicyConfig, PolicyTrainer, PretrainArguments
from harmonyrl.schema import Split

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/pretrain.yaml", type=Path)
	parser.add_argument("--name", help="run directory name, the start time by default")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(message)s")
	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
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
	data_dir = Path(config["data_dir"])
	policy = ChordPolicy(PolicyConfig(**config["model"]))
	trainer = PolicyTrainer(
		policy,
		PretrainArguments(output_dir=run_dir, seed=config["seed"], **config["train"]),
		train=read_segments(data_dir / f"segments_{Split.TRAIN.value}.jsonl"),
		valid=read_segments(data_dir / f"segments_{Split.VALID.value}.jsonl"),
	)

	logging.info("%d parameters", sum(p.numel() for p in policy.parameters()))

	best = trainer.train()
	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	meta["best"] = best
	_write_json(run_dir / "meta.json", meta)

	logging.info(
		"best epoch %d, valid NLL %.4f, in %s", best["epoch"], best["valid_nll"], run_dir / "best"
	)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
