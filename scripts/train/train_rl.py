"""Fine-tune the pretrained policy against the fixed preference reward or the local reward,
with a KL to it.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.data import read_segments
from harmonyrl.models import ChordPolicy, ChordReward
from harmonyrl.rl import RLArguments, RLTrainer, policy_from_reference
from harmonyrl.schema import Split

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="train_rl")
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
	train = read_segments(data_dir / f"segments_{Split.TRAIN.value}.jsonl")
	valid = read_segments(data_dir / f"segments_{Split.VALID.value}.jsonl")

	arguments = RLArguments(output_dir=run_dir, seed=config["seed"], **config["train"])
	reward = None
	if arguments.reward == "learned":
		reward = ChordReward.from_pretrained(config["reward_model"])

	reference = ChordPolicy.from_pretrained(config["reference"])
	trainer = RLTrainer(
		policy_from_reference(reference),
		reference,
		reward,
		arguments,
		train=train,
		valid=valid,
	)

	last = trainer.train()
	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	meta["last"] = last
	_write_json(run_dir / "meta.json", meta)

	logging.info("final policy in %s", run_dir / "final")


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
