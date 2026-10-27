"""Evaluate the harmonizations of 1 system on a data split against the human accompaniments.

Writes the config, git state, the metrics, and the harmonizations as segments into a new directory
named after the system, with the source tempos beside them so that renderings keep tempo.
"""

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from harmonyrl.data import read_segments, write_segments
from harmonyrl.metrics import evaluate_corpus
from harmonyrl.models import ChordHMM, ChordPolicy
from harmonyrl.schema import ChordEvent, Segment, Split

REPO = Path(__file__).resolve().parent.parent

type Harmonizer = Callable[[Sequence[Segment]], list[Sequence[ChordEvent]]]


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
	harmonizations = _system(config["systems"][args.system])(segments)
	write_segments(
		[
			Segment(s.id, s.split, s.melody, tuple(e))
			for s, e in zip(segments, harmonizations, strict=True)
		],
		run_dir / "harmonizations.jsonl",
	)

	tempos = data_dir / "tempos.json"
	if tempos.exists():
		shutil.copy(tempos, run_dir / "tempos.json")

	metrics = evaluate_corpus(segments, harmonizations, train)
	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	_write_json(run_dir / "meta.json", meta)
	_write_json(run_dir / "metrics.json", metrics)
	print(json.dumps(metrics, indent=2))


def _system(settings: dict[str, Any]) -> Harmonizer:
	"""How a system harmonizes segments, reading only their melodies except for `human`."""
	kind = settings["kind"]
	if kind == "human":
		return lambda segments: [segment.events for segment in segments]
	if kind == "hmm":
		hmm = ChordHMM.from_pretrained(settings["model"])
		return lambda segments: [hmm.harmonize(segment.melody) for segment in segments]
	if kind == "policy":
		return lambda segments: _generate(segments, settings)
	raise ValueError(f"unknown system kind {kind!r}")


def _generate(segments: Sequence[Segment], settings: dict[str, Any]) -> list[Sequence[ChordEvent]]:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	policy = ChordPolicy.from_pretrained(settings["model"]).to(device)

	return list(
		policy.generate(
			[segment.melody for segment in segments],
			temperature=settings.get("temperature", 1.0),
			greedy=settings.get("greedy", False),
			generator=torch.Generator(device).manual_seed(settings.get("seed", 0)),
			batch_size=settings["batch_size"],
		)
	)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
