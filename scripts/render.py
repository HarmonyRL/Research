"""Render segments to MIDI, WAV, and score formats with the fixed voicing."""

import json
import random
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
from harmonyrl.render import (
	melody_to_score,
	midi_to_wav,
	musicxml_to_formats,
	wav_to_mp3,
	write_midi,
	write_score,
)
from harmonyrl.schema import Segment

REPO = Path(__file__).resolve().parent.parent
_SUPPORTED_FORMATS = frozenset({"midi", "wav", "mp3", "musicxml", "pdf", "png", "mscz", "mscx"})
_SCORE_FORMATS = frozenset({"musicxml", "pdf", "png", "mscz", "mscx"})


@hydra.main(config_path="../configs", config_name="rendering")
def main(cfg: DictConfig) -> None:
	config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
	formats = config["formats"]
	unsupported = set(formats) - _SUPPORTED_FORMATS
	if unsupported:
		raise ValueError(f"unknown formats: {sorted(unsupported)}")

	paths = [Path(path) for path in config["segments"]]
	segments = [segment for path in paths for segment in read_segments(path)]
	selected = _select_segments(
		segments, ids=config["ids"], sample=config["sample"], seed=config["seed"]
	)

	output = Path(HydraConfig.get().runtime.output_dir)

	for segment in selected:
		_render_segment(segment, output=output, formats=formats, config=config)

	manifest = {
		"git": _git_hash(),
		"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
		"config": config,
		"segments": [segment.id for segment in selected]
	}
	(output / "manifest.json").write_text(
		json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(f"rendered {len(selected)} segments to {output}")


def _select_segments(
	segments: list[Segment],
	*,
	ids: list[str] | None,
	sample: int | None,
	seed: int | None,
) -> list[Segment]:
	if ids:
		wanted = set(ids)
		missing = wanted - {s.id for s in segments}
		if missing:
			raise ValueError(f"IDs not in the segment files: {sorted(missing)}")

		return [s for s in segments if s.id in wanted]

	if sample is None:
		raise ValueError("select segments with ids or sample and seed")

	if seed is None:
		raise ValueError("sample requires seed")

	return random.Random(seed).sample(segments, sample)


def _render_segment(
	segment: Segment,
	*,
	output: Path,
	formats: Sequence[str],
	config: dict[str, Any],
) -> None:
	format_set = set(formats)
	base = output / segment.id

	if "midi" in format_set or "wav" in format_set or "mp3" in format_set:
		write_midi(segment.melody, segment.events, base.with_suffix(".mid"))
	if "wav" in format_set or "mp3" in format_set:
		midi_to_wav(
			base.with_suffix(".mid"),
			base.with_suffix(".wav"),
			Path(config["soundfont"]).expanduser(),
			sample_rate=config["sample_rate"],
			gain=config["gain"],
		)
	if "mp3" in format_set:
		wav_to_mp3(
			base.with_suffix(".wav"), base.with_suffix(".mp3"), bitrate=config["mp3_bitrate"]
		)

	if not (format_set & _SCORE_FORMATS):
		return

	musicxml_path = base.with_suffix(".musicxml")
	write_score(melody_to_score(segment.melody, segment.events), musicxml_path)

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
