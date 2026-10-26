"""Evaluate the harmonizations of 1 system on a data split against the human accompaniments.

Writes the config, git state, and the metrics into a new directory named after the system.
"""

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from harmonyrl.data import read_segments
from harmonyrl.metrics import evaluate_corpus
from harmonyrl.models import ChordHMM
from harmonyrl.schema import ChordEvent, Segment, Split

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("system", help="a key of `systems` in the config")
	parser.add_argument("--config", default="configs/evaluate.yaml", type=Path)
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	started = datetime.now(UTC)
	run_dir = Path(config["run_dir"]) / args.system
	run_dir.mkdir(parents=True)

	meta: dict[str, Any] = {
		"git": _git("rev-parse", "--short", "HEAD"),
		"git_dirty": bool(_git("status", "--porcelain")),
		"started": started.isoformat(timespec="seconds"),
		"config": config,
		"system": args.system,
	}

	data_dir = Path(config["data_dir"])
	segments = read_segments(data_dir / f"segments_{Split(config['split']).value}.jsonl")
	train = read_segments(data_dir / f"segments_{Split.TRAIN.value}.jsonl")
	harmonize = _system(args.system, config["systems"][args.system])
	harmonizations = [harmonize(segment) for segment in segments]

	metrics = evaluate_corpus(segments, harmonizations, train)
	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	_write_json(run_dir / "meta.json", meta)
	_write_json(run_dir / "metrics.json", metrics)
	print(json.dumps(metrics, indent=2))


def _system(name: str, settings: dict[str, Any]) -> Callable[[Segment], Sequence[ChordEvent]]:
	"""How the named system harmonizes a segment, reading only its melody except for `human`."""
	if name == "human":
		return lambda segment: segment.events
	if name == "hmm":
		hmm = ChordHMM.from_pretrained(settings["model"])
		return lambda segment: hmm.harmonize(segment.melody)
	raise ValueError(f"unknown system {name!r}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
