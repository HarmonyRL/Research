"""Build a round of owner rankings: K continuations of the human chords of short segments."""

import csv
import json
import random
import string
import subprocess
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl_research.data import read_segments, write_segments
from harmonyrl_research.models import (
	ChordPolicy,
	ChordReward,
	best_of_group,
	color_groups,
	draw_groups,
	head_disagreement,
	policy_groups,
	ranking_groups,
	seconds,
)
from harmonyrl_research.render import (
	melody_to_score,
	midi_to_wav,
	musicxml_to_formats,
	write_midi,
	write_score,
)
from harmonyrl_research.rl import policy_from_reference
from harmonyrl_research.schema import Segment, Split, merge_repeats, no_chord_limit
from harmonyrl_research.tools import to_white_keys

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="build_round")
def main(cfg: DictConfig) -> None:
	config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
	started = datetime.now(ZoneInfo("Asia/Seoul"))
	run_dir = Path(HydraConfig.get().runtime.output_dir)
	if any(run_dir.iterdir()):
		raise FileExistsError(f"round directory {run_dir} is not empty")

	meta: dict[str, Any] = {
		"git": _git("rev-parse", "--short", "HEAD"),
		"git_dirty": bool(_git("status", "--porcelain")),
		"started": started.isoformat(timespec="seconds"),
		"config": config,
	}
	_write_json(run_dir / "meta.json", meta)

	betas, policies, draws = config["color_betas"], config["policies"], config["draws"]
	seed, low, high = config["seed"], *config["seconds"]
	if policies and draws > 1:
		k = draws
	elif policies:
		k = len(policies)
	elif betas:
		k = 1 + len(betas)
	else:
		k = config["items"]

	judged = _judged_songs(run_dir.parent)
	segments = read_segments(
		Path(config["data_dir"]) / f"segments_{Split(config['split']).value}.jsonl"
	)

	pool = [
		s
		for s in segments
		if s.id not in judged and low <= seconds(s.melody) <= high and _keeps_no_chord_rule(s)
	]
	rng = random.Random(seed)
	rng.shuffle(pool)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	policy = policy_from_reference(ChordPolicy.from_pretrained(config["reference"])).to(device)
	generator = torch.Generator(device).manual_seed(seed)

	reward = None
	if config["reward_model"] is not None:
		reward = ChordReward.from_pretrained(config["reward_model"]).to(device)

	if policies:
		loaded = [
			(
				policy_from_reference(ChordPolicy.from_pretrained(p["model"])).to(device),
				p["temperature"],
			)
			for p in policies
		]
		if draws > 1:
			drawn = draw_groups(
				loaded,
				pool,
				config["groups"] // len(policies),
				draws,
				config["min_bars_apart"],
				generator,
			)
		else:
			drawn = policy_groups(
				loaded, pool, config["groups"], config["min_bars_apart"], generator
			)
	elif betas:
		drawn = color_groups(
			policy,
			pool,
			config["groups"],
			betas,
			config["best_of"],
			config["temperature"],
			config["min_bars_apart"],
			generator,
		)
	else:
		drawn = ranking_groups(
			policy,
			pool,
			config["groups"] * config["candidates"],
			k,
			config["temperature"],
			config["min_bars_apart"],
			generator,
		)

	disagreement: list[float | None] = [None] * len(drawn)

	if reward is not None:
		scores = head_disagreement(reward, drawn)
		order = sorted(range(len(drawn)), key=lambda i: scores[i], reverse=True)
		drawn = [drawn[i] for i in order]
		disagreement = [scores[i] for i in order]

	kept, unlabeled = drawn[: config["groups"]], drawn[config["groups"] :]
	best_of: list[bool] = [False] * len(kept)

	if reward is not None:
		for i in range(round(config["best_of_share"] * len(kept))):
			group = best_of_group(
				policy,
				reward,
				kept[i],
				config["best_of"],
				config["temperature"],
				config["min_bars_apart"],
				generator,
			)
			if group is not None:
				kept[i], best_of[i] = group, True

	order = list(range(len(kept)))
	rng.shuffle(order)
	kept = [kept[i] for i in order]
	best_of = [best_of[i] for i in order]
	disagreement = [disagreement[i] for i in order]

	letters = string.ascii_lowercase[:k]
	for i, letter in enumerate(letters):
		write_segments(
			[Segment(g.segment.id, g.segment.split, g.segment.melody, g.items[i]) for g in kept],
			run_dir / f"{letter}.jsonl",
		)
		write_segments(
			[
				Segment(g.segment.id, g.segment.split, g.segment.melody, g.items[i])
				for g in unlabeled
			],
			run_dir / f"unlabeled-{letter}.jsonl",
		)

	songs: dict[str, dict[str, Any]] = {}
	for row, group in enumerate(kept):
		group_id = f"g{row + 1:02d}"
		bar = group.segment.melody.meter.bar_eighths

		for letter, items in zip(letters, group.items, strict=True):
			shift = _render(
				Segment(group.segment.id, group.segment.split, group.segment.melody, items),
				run_dir / f"{group_id}{letter}",
				config,
			)

		songs[group_id] = {
			"id": group.segment.id,
			"row": row,
			"shift": shift,
			"seconds": round(seconds(group.segment.melody), 1),
			"bars": group.segment.melody.length / bar,
			"prefix_events": group.prefix,
			"split_bar": sum(e.duration for e in group.segment.events[: group.prefix]) // bar + 1,
			"best_of": letters[0] if best_of[row] else None,
			"disagreement": disagreement[row],
			"policy": group.policy,
		}

	group_ids = list(songs)
	repeats = rng.sample(group_ids[:10], min(config["repeats"], len(group_ids[:10])))
	listed = list(group_ids)

	for i, group_id in enumerate(repeats, start=1):
		listed.insert(rng.randrange(max(0, len(listed) - 9), len(listed) + 1), f"r{i}-{group_id}")

	with (run_dir / "groups.csv").open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["group_id", "item_id", "path", "tag", "bars"])

		for group_id in listed:
			name = group_id.split("-")[-1]

			for letter in letters:
				path = run_dir / f"{name}{letter}.wav"
				bars = ceil(songs[name]["bars"])
				writer.writerow([group_id, letter, path.resolve(), "", bars])

	_write_json(
		run_dir / "key.json",
		{
			"seed": seed,
			"temperature": config["temperature"],
			"split": config["split"],
			"k": k,
			"betas": betas,
			"policies": policies,
			"draws": draws,
			"events": {letter: f"{letter}.jsonl" for letter in letters},
			"songs": songs,
			"unlabeled": {
				"events": {letter: f"unlabeled-{letter}.jsonl" for letter in letters},
				"groups": len(unlabeled),
			},
		},
	)

	meta["finished"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
	meta["groups"] = len(kept)
	meta["best_of_groups"] = sum(best_of)
	meta["unlabeled_groups"] = len(unlabeled)
	_write_json(run_dir / "meta.json", meta)
	print(
		f"{len(kept)} groups, {len(listed)} with repeats, {len(unlabeled)} unlabeled, in {run_dir}"
	)


def _render(segment: Segment, base: Path, config: dict[str, Any]) -> int:
	"""Render `segment` for the owner at `base`, on the white keys. Returns the shift."""
	listened, shift = to_white_keys(segment)
	events = merge_repeats(listened.events, listened.melody)
	midi, musicxml = base.with_suffix(".mid"), base.with_suffix(".musicxml")

	write_midi(listened.melody, events, midi)
	midi_to_wav(
		midi,
		base.with_suffix(".wav"),
		Path(config["soundfont"]).expanduser(),
		sample_rate=config["sample_rate"],
		gain=config["gain"],
	)
	write_score(melody_to_score(listened.melody, events), musicxml)
	musicxml_to_formats(musicxml, [base.with_suffix(".png")], Path(config["mscore"]).expanduser())

	return shift


def _judged_songs(root: Path) -> set[str]:
	"""Segment ids of the songs of every round under `root`."""
	return {
		song["id"]
		for key in root.glob("*/key.json")
		for song in json.loads(key.read_text(encoding="utf-8"))["songs"].values()
	}


def _keeps_no_chord_rule(segment: Segment) -> bool:
	"""Whether no chord sounds only as the 1st event, within the `no_chord_limit` of the melody."""
	first = segment.events[0]
	return all(e.chord is not None for e in segment.events[1:]) and (
		first.chord is not None or first.duration <= no_chord_limit(segment.melody)
	)


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	main()
