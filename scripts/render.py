"""Render segments to MIDI, WAV, and score formats with the fixed voicing.

Selects segments from JSONL files by ID or by seeded random sample and writes the
requested formats under runs/render/<name>/, plus a manifest with the configuration,
git hash, and the tempo used per segment.
"""

import argparse
import json
import random
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from harmonyrl.data import read_segments
from harmonyrl.render import (
	DEFAULT_TEMPO,
	melody_to_score,
	midi_to_wav,
	musicxml_to_formats,
	write_midi,
	write_score,
)
from harmonyrl.schema import Segment

REPO = Path(__file__).resolve().parent.parent
_SUPPORTED_FORMATS = frozenset({"midi", "wav", "musicxml", "pdf", "png", "mscz", "mscx"})
_SCORE_FORMATS = frozenset({"musicxml", "pdf", "png", "mscz", "mscx"})


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/rendering.yaml", type=Path)
	parser.add_argument(
		"--segments",
		nargs="+",
		default=[
			Path(f"data/processed/segments_{split}.jsonl") for split in ("train", "valid", "test")
		],
		type=Path,
	)
	parser.add_argument("--ids", help="comma-separated segment IDs")
	parser.add_argument("--sample", type=int, help="random sample size (requires --seed)")
	parser.add_argument("--seed", type=int)
	parser.add_argument("--formats", default="midi,wav,musicxml,pdf", help="comma-separated")
	parser.add_argument("--name", required=True, help="output directory name under output_root")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	formats = args.formats.split(",")
	unsupported = set(formats) - _SUPPORTED_FORMATS
	if unsupported:
		parser.error(f"unknown formats: {sorted(unsupported)}")

	segments = [segment for path in args.segments for segment in read_segments(path)]
	selected = _select_segments(parser, segments, ids=args.ids, sample=args.sample, seed=args.seed)

	output = Path(config["output_root"]) / args.name
	output.mkdir(parents=True, exist_ok=True)
	tempos = _load_tempos(args.segments)

	rendered: dict[str, int] = {}
	for segment in selected:
		tempo = tempos.get(segment.id, config.get("default_tempo", DEFAULT_TEMPO))
		_render_segment(segment, output=output, formats=formats, tempo=tempo, config=config)
		rendered[segment.id] = tempo

	manifest = {
		"git": _git_hash(),
		"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
		"config": config,
		"segments_files": [str(path) for path in args.segments],
		"formats": formats,
		"selection": {"ids": args.ids, "sample": args.sample, "seed": args.seed},
		"tempos": rendered,
	}
	(output / "manifest.json").write_text(
		json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(f"rendered {len(rendered)} segments to {output}")


def _select_segments(
	parser: argparse.ArgumentParser,
	segments: list[Segment],
	*,
	ids: str | None,
	sample: int | None,
	seed: int | None,
) -> list[Segment]:
	if ids:
		wanted = set(ids.split(","))
		missing = wanted - {s.id for s in segments}
		if missing:
			parser.error(f"IDs not in the segment files: {sorted(missing)}")
		return [s for s in segments if s.id in wanted]
	if sample is None:
		parser.error("select segments with --ids or --sample/--seed")
	if seed is None:
		parser.error("--sample requires --seed")
	return random.Random(seed).sample(segments, sample)


def _render_segment(
	segment: Segment,
	*,
	output: Path,
	formats: Sequence[str],
	tempo: int,
	config: dict[str, Any],
) -> None:
	format_set = set(formats)
	base = output / segment.id
	if "midi" in format_set or "wav" in format_set:
		write_midi(segment.melody, segment.events, base.with_suffix(".mid"), tempo)
	if "wav" in format_set:
		midi_to_wav(
			base.with_suffix(".mid"),
			base.with_suffix(".wav"),
			Path(config["soundfont"]).expanduser(),
			sample_rate=config["sample_rate"],
			gain=config["gain"],
		)
	if not (format_set & _SCORE_FORMATS):
		return

	musicxml_path = base.with_suffix(".musicxml")
	write_score(melody_to_score(segment.melody, segment.events, tempo=tempo), musicxml_path)
	targets: list[Path] = []
	if "pdf" in format_set:
		targets.append(base.with_suffix(".pdf"))
	if "png" in format_set:
		targets.append(base.with_suffix(".png"))
	if "mscz" in format_set:
		targets.append(base.with_suffix(".mscz"))
	if "mscx" in format_set:
		directory = output / f"{segment.id}.mscx.d"
		directory.mkdir(exist_ok=True)
		targets.append(directory / f"{segment.id}.mscx")
	if targets:
		musicxml_to_formats(musicxml_path, targets, Path(config["mscore"]).expanduser())


def _load_tempos(segments_paths: list[Path]) -> dict[str, int]:
	tempos_path = segments_paths[0].parent / "tempos.json"
	if not tempos_path.exists():
		return {}
	return json.loads(tempos_path.read_text(encoding="utf-8"))


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
