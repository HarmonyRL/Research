"""Render segments to MIDI, WAV, and score formats with the fixed voicing.

Selects segments from a JSONL file by ID or by seeded random sample and writes the
requested formats under runs/render/<name>/, plus a manifest with the configuration,
git hash, and the tempo used per segment.
"""

import argparse
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from harmonyrl.data import read_segments
from harmonyrl.render import (
	DEFAULT_TEMPO,
	midi_to_wav,
	musicxml_to_formats,
	segment_to_score,
	write_midi,
	write_score,
)

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/render.yaml", type=Path)
	parser.add_argument("--segments", default="data/processed/segments_train.jsonl", type=Path)
	parser.add_argument("--ids", help="comma-separated segment IDs")
	parser.add_argument("--sample", type=int, help="random sample size (requires --seed)")
	parser.add_argument("--seed", type=int)
	parser.add_argument("--formats", default="midi,wav,musicxml,pdf", help="comma-separated")
	parser.add_argument("--name", required=True, help="output directory name under output_root")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	formats = args.formats.split(",")
	unsupported = set(formats) - {"midi", "wav", "musicxml", "pdf", "mscz", "mscx"}
	if unsupported:
		parser.error(f"unknown formats: {sorted(unsupported)}")

	segments = read_segments(args.segments)
	if args.ids:
		wanted = set(args.ids.split(","))
		missing = wanted - {s.id for s in segments}
		if missing:
			parser.error(f"IDs not in {args.segments}: {sorted(missing)}")
		selected = [s for s in segments if s.id in wanted]
	elif args.sample is not None:
		if args.seed is None:
			parser.error("--sample requires --seed")
		selected = random.Random(args.seed).sample(segments, args.sample)
	else:
		parser.error("select segments with --ids or --sample/--seed")

	output = Path(config["output_root"]) / args.name
	output.mkdir(parents=True, exist_ok=True)
	tempos = _load_tempos(args.segments)
	soundfont = Path(config["soundfont"]).expanduser()

	rendered = {}
	for segment in selected:
		tempo = tempos.get(segment.id, config.get("default_tempo", DEFAULT_TEMPO))
		base = output / segment.id
		if "midi" in formats or "wav" in formats:
			write_midi(segment, base.with_suffix(".mid"), tempo)
		if "wav" in formats:
			midi_to_wav(
				base.with_suffix(".mid"),
				base.with_suffix(".wav"),
				soundfont,
				sample_rate=config["sample_rate"],
				gain=config["gain"],
			)
		targets = {}
		if {"musicxml", "pdf", "mscz", "mscx"} & set(formats):
			write_score(segment_to_score(segment), base.with_suffix(".musicxml"))
		if "pdf" in formats:
			targets["pdf"] = base.with_suffix(".pdf")
		if "mscz" in formats:
			targets["mscz"] = base.with_suffix(".mscz")
		if "mscx" in formats:
			directory = output / f"{segment.id}.mscx.d"
			directory.mkdir(exist_ok=True)
			targets["mscx"] = directory / f"{segment.id}.mscx"
		if targets:
			musicxml_to_formats(
				base.with_suffix(".musicxml"), targets, Path(config["mscore"]).expanduser()
			)
		rendered[segment.id] = tempo

	manifest = {
		"git": _git_hash(),
		"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
		"config": config,
		"segments_file": str(args.segments),
		"formats": formats,
		"selection": {"ids": args.ids, "sample": args.sample, "seed": args.seed},
		"tempos": rendered,
	}
	(output / "manifest.json").write_text(
		json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(f"rendered {len(rendered)} segments to {output}")


def _load_tempos(segments_path: Path) -> dict[str, int]:
	tempos_path = segments_path.parent / "tempos.json"
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
