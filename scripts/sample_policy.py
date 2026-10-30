"""Sample a harmonization of every melody in some data splits from a chord policy.

Writes the config, git state, and per split the samples as segments into a new run directory, with
the source tempos beside them. Critic training reads the samples as negatives.
"""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from harmonyrl.data import read_segments, write_segments
from harmonyrl.models import ChordPolicy
from harmonyrl.rl import policy_from_reference
from harmonyrl.schema import Segment, Split

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/sample_policy.yaml", type=Path)
	parser.add_argument("--name", help="run directory name, the start time by default")
	args = parser.parse_args()

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

	device = "cuda" if torch.cuda.is_available() else "cpu"

	policy = ChordPolicy.from_pretrained(config["model"])
	if config["restrict_no_chord"]:
		policy = policy_from_reference(policy).eval()
	policy = policy.to(device)

	generator = torch.Generator(device).manual_seed(config["seed"])
	data_dir = Path(config["data_dir"])
	for split in map(Split, config["splits"]):
		segments = read_segments(data_dir / f"segments_{split.value}.jsonl")
		samples = policy.generate(
			[s.melody for s in segments],
			temperature=config["temperature"],
			generator=generator,
			batch_size=config["batch_size"],
		)
		write_segments(
			[Segment(s.id, s.split, s.melody, e) for s, e in zip(segments, samples, strict=True)],
			run_dir / f"segments_{split.value}.jsonl",
		)

	tempos = data_dir / "tempos.json"
	if tempos.exists():
		shutil.copy(tempos, run_dir / "tempos.json")

	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	_write_json(run_dir / "meta.json", meta)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
