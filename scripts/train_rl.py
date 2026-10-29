"""Fine-tune the pretrained policy against the fixed structure critic, with a KL to it.

Writes the config, git state, per-update metrics, and policy checkpoints into a new run directory.
"""

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from harmonyrl.data import read_segments
from harmonyrl.models import ChordCritic, ChordPolicy
from harmonyrl.rl import RLArguments, RLTrainer, policy_from_reference
from harmonyrl.schema import Split

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/train_rl.yaml", type=Path)
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

	data_dir = Path(config["data_dir"])
	train = read_segments(data_dir / f"segments_{Split.TRAIN.value}.jsonl")
	valid = read_segments(data_dir / f"segments_{Split.VALID.value}.jsonl")

	reference = ChordPolicy.from_pretrained(config["reference"])
	trainer = RLTrainer(
		policy_from_reference(reference),
		reference,
		ChordCritic.from_pretrained(config["critic"]),
		RLArguments(output_dir=run_dir, seed=config["seed"], **config["train"]),
		train=train,
		valid=valid,
	)

	last = trainer.train()
	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
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
