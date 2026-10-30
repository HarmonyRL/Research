"""Evaluate the harmonizations of 1 system on a data split against the human accompaniments.

Writes the config, git state, the metrics, and each sample of the harmonizations as segments into a
new directory named after the system, with the source tempos beside them so that renderings keep
tempo.
"""

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from harmonyrl.data import read_segments, write_segments
from harmonyrl.metrics import evaluate_corpus
from harmonyrl.models import ChordCritic, ChordHMM, ChordPolicy
from harmonyrl.rl import policy_from_reference, terminal_rewards
from harmonyrl.schema import ChordEvent, Segment, Split

REPO = Path(__file__).resolve().parent.parent

type Harmonizer = Callable[[Sequence[Segment]], list[list[Sequence[ChordEvent]]]]


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
	for i in range(len(harmonizations[0])):
		write_segments(
			[
				Segment(s.id, s.split, s.melody, tuple(samples[i]))
				for s, samples in zip(segments, harmonizations, strict=True)
			],
			run_dir / f"harmonizations-{i}.jsonl",
		)

	tempos = data_dir / "tempos.json"
	if tempos.exists():
		shutil.copy(tempos, run_dir / "tempos.json")

	metrics = evaluate_corpus(segments, harmonizations, train)
	metrics |= _critic_logits(config["critics"], segments, harmonizations)
	meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
	_write_json(run_dir / "meta.json", meta)
	_write_json(run_dir / "metrics.json", metrics)
	print(json.dumps(metrics, indent=2))


def _system(settings: dict[str, Any]) -> Harmonizer:
	"""How a system harmonizes segments, reading only their melodies except for `human`."""
	kind = settings["kind"]
	if kind == "human":
		return lambda segments: [[segment.events] for segment in segments]
	if kind == "hmm":
		hmm = ChordHMM.from_pretrained(settings["model"])
		return lambda segments: [[hmm.harmonize(segment.melody)] for segment in segments]
	if kind == "policy":
		return lambda segments: _generate(segments, settings, settings.get("samples", 1))
	if kind == "rerank":
		return lambda segments: _rerank(segments, settings)
	raise ValueError(f"unknown system kind {kind!r}")


def _generate(
	segments: Sequence[Segment], settings: dict[str, Any], samples: int
) -> list[list[Sequence[ChordEvent]]]:
	"""`samples` harmonizations of each segment from the policy, those of 1 segment in a run."""
	device = _device()
	policy = ChordPolicy.from_pretrained(settings["model"])
	if settings.get("restrict_no_chord", False):
		policy = policy_from_reference(policy).eval()

	events = policy.to(device).generate(
		[segment.melody for segment in segments for _ in range(samples)],
		temperature=settings.get("temperature", 1.0),
		greedy=settings.get("greedy", False),
		generator=torch.Generator(device).manual_seed(settings.get("seed", 0)),
		batch_size=settings["batch_size"],
	)

	return [list(events[i : i + samples]) for i in range(0, len(events), samples)]


def _rerank(
	segments: Sequence[Segment], settings: dict[str, Any]
) -> list[list[Sequence[ChordEvent]]]:
	"""Per sample, 1 of `candidates` new policy samples, chosen by the clipped critic logit.

	With `beta` 0 the best candidate is chosen; otherwise 1 is drawn in proportion to
	exp(reward / beta), which approximates sampling the target of the RL objective.
	"""
	samples, candidates, beta = settings["samples"], settings["candidates"], settings["beta"]
	if beta < 0:
		raise ValueError(f"beta must not be negative, got {beta}")

	drawn = _generate(segments, settings, samples * candidates)
	critic = ChordCritic.from_pretrained(settings["critic"]).to(_device())
	rewards = terminal_rewards(
		critic,
		[s.melody for s, group in zip(segments, drawn, strict=True) for _ in group],
		[events for group in drawn for events in group],
		settings["clip_reward"],
	)
	rewards = rewards.cpu().view(len(segments), samples, candidates)

	if beta == 0:
		chosen = rewards.argmax(dim=-1)
	else:
		weights = (rewards / beta).softmax(dim=-1).view(-1, candidates)
		generator = torch.Generator().manual_seed(settings.get("seed", 0))
		chosen = torch.multinomial(weights, 1, generator=generator).view(len(segments), samples)

	return [
		[group[k * candidates + c] for k, c in enumerate(row)]
		for group, row in zip(drawn, chosen.tolist(), strict=True)
	]


def _critic_logits(
	critics: dict[str, str],
	segments: Sequence[Segment],
	harmonizations: Sequence[Sequence[Sequence[ChordEvent]]],
) -> dict[str, float]:
	"""Mean unclipped logit of every harmonization under each critic of `critics`."""
	melodies = [
		s.melody for s, samples in zip(segments, harmonizations, strict=True) for _ in samples
	]
	events = [e for samples in harmonizations for e in samples]
	logits: dict[str, float] = {}
	for name, path in critics.items():
		critic = ChordCritic.from_pretrained(path).to(_device())
		logits[f"{name}_logit"] = float(np.mean(critic.score(melodies, events)))
	return logits


def _device() -> str:
	return "cuda" if torch.cuda.is_available() else "cpu"


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
