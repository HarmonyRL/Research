"""Pretrain the chord policy π_ref on human accompaniments with teacher forcing."""

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

from harmonyrl_research.data import read_segments
from harmonyrl_research.models import ChordPolicy, PolicyConfig, PolicyTrainer, PretrainArguments
from harmonyrl_research.schema import Split

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="pretrain")
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
	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
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
