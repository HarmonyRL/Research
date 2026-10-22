"""Prepare the contract dataset from the two source files.

Reads the simplified and raw Hooktheory files, converts them into Segments, and writes
one JSONL file per split plus a meta.json with every count the pipeline produced.
"""

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from harmonyrl.data import (
	alignment_report,
	load_json_gz,
	prepare_dataset,
	usable_ids,
	write_segments,
)
from harmonyrl.schema import Segment, Split, project

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/prepare_data.yaml", type=Path)
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	source_dir = Path(config["source_dir"]).expanduser()
	output_dir = Path(config["output_dir"])
	output_dir.mkdir(parents=True, exist_ok=True)

	simple = load_json_gz(source_dir / config["simplified_file"])
	raw = load_json_gz(source_dir / config["raw_file"])
	print(f"loaded {len(simple)} simplified, {len(raw)} raw items")

	segments, report = prepare_dataset(simple, raw)
	used_ids = usable_ids(simple, raw)[0]
	align = alignment_report(simple, raw, used_ids)

	by_split: dict[Split, list[Segment]] = {split: [] for split in Split}
	for segment in segments:
		by_split[segment.split].append(segment)
	for split, group in by_split.items():
		write_segments(group, output_dir / f"segments_{split.value}.jsonl")

	def meter_of(sid: str) -> str:
		entry = simple[sid]["annotations"]["meters"][0]
		return f"{entry['beats_per_bar']}/{entry['beat_unit']}"

	meta = {
		"git": _git_hash(),
		"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
		"config": config,
		"filtering": {
			"total": report.total,
			"both_present": report.both_present,
			"meter_change_excluded": report.meter_change,
			"used": report.used,
			"with_modulation": report.with_modulation,
			"dropped_unrepresentable": len(report.dropped_no_chord_overlong)
			+ len(report.items_without_keys),
			"meter_distribution": dict(
				sorted(Counter(map(meter_of, used_ids)).items(), key=lambda kv: -kv[1])
			),
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
			"no_chord_collapsed": report.no_chord_collapsed,
			"key_starts_snapped": report.key_starts_snapped,
			"key_spans_dropped_at_end": report.key_spans_dropped_at_end,
			"key_spans_merged_short": report.key_spans_merged_short,
		},
		"projection": _projection(segments),
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
