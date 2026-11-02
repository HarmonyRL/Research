"""Compute baseline dataset statistics.

2 levels of statistics over the prepared segments:
- melody and chord statistics on the final segments (non-chord tones, chord kinds,
  barline-crossing chords), for the 4/4 subset and for every x/4 meter;
- grid statistics on the source annotations (chord onsets off the half-bar grid and
  chords lost when quantized to coarser grids), which need exact beat positions.
"""

import json
import subprocess
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from harmonyrl.data import beat_value, load_json_gz, read_segments
from harmonyrl.metrics import idiom_counts
from harmonyrl.schema import Segment

REPO = Path(__file__).resolve().parent.parent
GRIDS = {"half_bar": Fraction(2), "beat": Fraction(1), "eighth": Fraction(1, 2)}
"""Quantization grids in beats, coarser first."""


@hydra.main(config_path="../configs", config_name="stats")
def main(cfg: DictConfig) -> None:
	config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
	processed = Path(config["processed_dir"])
	segments = [
		segment
		for name in ("train", "valid", "test")
		for segment in read_segments(processed / f"segments_{name}.jsonl")
	]
	four4 = [
		s for s in segments if (s.melody.meter.beats_per_bar, s.melody.meter.beat_unit) == (4, 4)
	]
	x4 = [s for s in segments if s.melody.meter.beat_unit == 4]

	stats: dict[str, Any] = {
		"git": _git_hash(),
		"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
		"config": config,
		"counts": {"segments": len(segments), "four4": len(four4), "x4": len(x4)},
		"grid": _grid_stats(config, [s.id for s in four4]),
		"four4": _segment_stats(four4),
		"x4": {
			**_segment_stats(x4),
			"barline_crossing_pct": _round(*_barline_crossing(x4)),
		},
	}

	(Path(HydraConfig.get().runtime.output_dir) / "stats.json").write_text(
		json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(yaml.safe_dump(_summary(stats), sort_keys=False, allow_unicode=True))


def _segment_stats(segments: list[Segment]) -> dict[str, Any]:
	counts = Counter[str]()
	for segment in segments:
		counts += idiom_counts(segment.melody, segment.events, projected=False)
	return {
		"segments": len(segments),
		"notes": counts["notes"],
		"non_chord_tone_pct": _round(counts["non_chord_tones"], counts["notes_under_chord"]),
		"long_non_chord_note_pct": _round(counts["long_non_chord_tones"], counts["notes"]),
		"long_non_chord_segment_pct": _round(counts["segments_with_long"], len(segments)),
		"resolution_pct": _round(counts["resolved"], counts["long_non_chord_tones"]),
		"chord_kind_pct": _round(counts["tension_chords"], counts["chords"]),
		"borrowed_pct": _round(counts["borrowed_chords"], counts["chords"]),
		"applied_pct": _round(counts["applied_chords"], counts["chords"]),
	}


def _barline_crossing(segments: list[Segment]) -> tuple[int, int]:
	"""Count of chord events spanning a barline and the total chord events."""
	total = crossing = 0
	for segment in segments:
		bar = segment.melody.meter.bar_eighths
		position = 0
		for event in segment.events:
			if event.chord is not None:
				total += 1
				crossing += position // bar != (position + event.duration - 1) // bar
			position += event.duration
	return crossing, total


def _grid_stats(config: dict[str, Any], ids: list[str]) -> dict[str, Any]:
	"""Chord onset statistics on the source annotations, in exact beats."""
	source = Path(config["source_dir"]).expanduser()
	raw = load_json_gz(source / config["raw_file"])
	total = off_half = 0
	lost = dict.fromkeys(GRIDS, 0)
	for sid in ids:
		chords = raw[sid]["json"].get("chords") or ()
		spans = []
		for chord in chords:
			if chord.get("isRest"):
				continue
			onset = beat_value(chord["beat"]) - 1
			spans.append((onset, onset + beat_value(chord["duration"])))
		total += len(spans)
		off_half += sum(onset % 2 != 0 for onset, _ in spans)
		for name, grid in GRIDS.items():
			lost[name] += sum(_snap(end, grid) <= _snap(onset, grid) for onset, end in spans)
	return {
		"chords": total,
		"off_half_bar_pct": _round(off_half, total),
		"grid_loss_pct": {name: _round(count, total) for name, count in lost.items()},
	}


def _snap(value: Fraction, grid: Fraction) -> Fraction:
	return (value / grid + Fraction(1, 2)).__floor__() * grid


def _round(numerator: int, denominator: int) -> float:
	return round(float(Fraction(numerator, denominator) * 100), 3) if denominator else 0.0


def _summary(stats: dict[str, Any]) -> dict[str, Any]:
	return {key: stats[key] for key in ("counts", "grid", "four4", "x4")}


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
