"""Sample a harmonization of every melody in some data splits from a chord policy."""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.data import read_segments, write_segments
from harmonyrl.models import ChordPolicy
from harmonyrl.rl import policy_from_reference
from harmonyrl.schema import Segment, Split

REPO = Path(__file__).resolve().parent.parent


@hydra.main(config_path="../configs", config_name="sample_policy")
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

	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	_write_json(run_dir / "meta.json", meta)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
