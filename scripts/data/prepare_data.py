"""Prepare the contract dataset from the two source files."""

import json
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl_research.data import (
	alignment_report,
	load_json_gz,
	prepare_dataset,
	write_segments,
)
from harmonyrl_research.schema import Segment, Split, project

REPO = Path(__file__).resolve().parents[2]


@hydra.main(config_path="../../configs", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
	config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
	source_dir = Path(config["source_dir"]).expanduser()
	output_dir = Path(HydraConfig.get().runtime.output_dir)

	simple = load_json_gz(source_dir / config["simplified_file"])
	raw = load_json_gz(source_dir / config["raw_file"])
	print(f"loaded {len(simple)} simplified, {len(raw)} raw items")

	segments, report = prepare_dataset(simple, raw)
	align = alignment_report(simple, raw, report.used_ids)

	by_split: dict[Split, list[Segment]] = {split: [] for split in Split}
	for segment in segments:
		by_split[segment.split].append(segment)
	for split, group in by_split.items():
		write_segments(group, output_dir / f"segments_{split.value}.jsonl")

	meta = {
		"git": _git_hash(),
		"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
		"config": config,
		"filtering": {
			"total": report.total,
			"both_present": report.both_present,
			"meter_change_excluded": report.meter_change,
			"used": report.used,
			"with_modulation": report.with_modulation,
			"dropped_unrepresentable": len(report.dropped_no_chord_overlong)
			+ len(report.items_without_keys),
			"meter_distribution": report.meter_distribution,
		},
		"segments": {split.value: len(group) for split, group in by_split.items()}
		| {"total": len(segments)},
		"alignment": {
			"compared": align.compared,
			"equal": align.equal,
			"differing": align.differing,
		},
		"conversion": {
			"items_without_keys": report.items_without_keys,
			"length_trimmed": report.length_trimmed,
			"dropped_no_chord_overlong": report.dropped_no_chord_overlong,
			"root_zero_as_no_chord": report.root_zero_as_no_chord,
			"borrowed_unknown": report.borrowed_unknown,
			"overlaps_clipped": report.overlaps_clipped,
			"chords_overlapped_away": report.chords_overlapped_away,
			"chords_lost_quantized": report.chords_lost_quantized,
			"key_starts_snapped": report.key_starts_snapped,
			"key_spans_dropped_at_end": report.key_spans_dropped_at_end,
		},
		"projection": _projection(segments),
		"tempos": {
			"known": sum(s.melody.tempo is not None for s in segments),
			"unknown": sum(s.melody.tempo is None for s in segments),
		},
	}
	(output_dir / "meta.json").write_text(
		json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)

	summary = (
		meta["filtering"]
		| {f"segments_{name}": count for name, count in meta["segments"].items()}
		| meta["projection"]
		| {
			"alignment_equal": align.equal,
			"alignment_differing": len(align.differing),
			**{f"conv_{k}": v for k, v in meta["conversion"].items()},
		}
	)
	print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


def _projection(segments: list[Segment]) -> dict[str, Any]:
	"""Share of chord events whose chord survives the vocabulary projection unchanged."""
	total = 0
	projected = 0
	for segment in segments:
		position = Fraction(0)
		for event in segment.events:
			key = segment.melody.key_at(position)
			if event.chord is not None:
				total += 1
				projected += project(event.chord, key) != event.chord
			position += event.duration
	ratio = projected / total if total else 0.0
	return {"chord_events": total, "projected": projected, "ratio": round(ratio, 6)}


def _git_hash() -> str:
	return subprocess.run(
		["git", "rev-parse", "--short", "HEAD"],
		cwd=REPO,
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()


if __name__ == "__main__":
	main()
