"""Fit the preference reward to the preference comparisons."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.models import (
	ChordReward,
	RewardArguments,
	RewardConfig,
	RewardTrainer,
	read_comparisons,
	read_unlabeled,
)

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="train_reward")
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

	comparisons = [c for round_dir in config["rounds"] for c in read_comparisons(Path(round_dir))]
	unlabeled = [p for round_dir in config["rounds"] for p in read_unlabeled(Path(round_dir))]

	torch.manual_seed(config["seed"])
	reward = ChordReward.from_backbone(RewardConfig(**config["model"]))
	trainer = RewardTrainer(
		reward,
		RewardArguments(output_dir=run_dir, seed=config["seed"], **config["train"]),
		comparisons,
		unlabeled,
	)

	metrics = trainer.train()
	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	meta["comparisons"] = len(comparisons)
	meta["unlabeled_pairs"] = len(unlabeled)
	meta["metrics"] = metrics
	_write_json(run_dir / "meta.json", meta)

	logging.info("reward in %s", run_dir / "model")


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
