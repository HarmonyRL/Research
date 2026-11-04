"""Fit the HMM baseline on the training split."""

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.data import read_segments
from harmonyrl.models import ChordHMM, HMMConfig
from harmonyrl.schema import ChordEvent, Melody, RawChord, Segment, Split, project

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="hmm")
def main(cfg: DictConfig) -> None:
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

	hmm = ChordHMM.fit(train, HMMConfig(**config["model"]))
	hmm.save_pretrained(run_dir / "model")

	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	meta["valid_frame_accuracy"] = _frame_accuracy(hmm, valid)
	_write_json(run_dir / "meta.json", meta)
	print(f"valid frame accuracy {meta['valid_frame_accuracy']:.4f}, saved to {run_dir / 'model'}")


def _frame_accuracy(hmm: ChordHMM, segments: Sequence[Segment]) -> float:
	"""Share of 8th notes whose projected chord equals the human one."""
	matches = 0
	total = 0

	for segment in segments:
		predicted = _frame_chords(segment.melody, hmm.harmonize(segment.melody))
		human = _frame_chords(segment.melody, segment.events)
		matches += sum(a == b for a, b in zip(predicted, human, strict=True))
		total += len(human)

	return matches / total


def _frame_chords(melody: Melody, events: Sequence[ChordEvent]) -> list[RawChord | None]:
	chords: list[RawChord | None] = []
	for event in events:
		chord = None if event.chord is None else project(event.chord, melody.key_at(len(chords)))
		chords.extend([chord] * event.duration)

	return chords


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
