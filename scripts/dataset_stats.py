"""Compute baseline dataset statistics.

Two levels of statistics over the prepared segments:
- melody and chord statistics on the final segments (non-chord tones, chord kinds,
  barline-crossing chords), for the 4/4 subset and for every x/4 meter;
- grid statistics on the source annotations (chord onsets off the half-bar grid and
  chords lost when quantized to coarser grids), which need exact beat positions.

Writes a stats.json with the configuration and git hash, and prints a summary.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from harmonyrl.data import beat_value, load_json_gz, read_segments
from harmonyrl.schema import ChordEvent, Melody, Note, RawChord, Segment, chord_pitch_classes

REPO = Path(__file__).resolve().parent.parent
GRIDS = {"half_bar": Fraction(2), "beat": Fraction(1), "eighth": Fraction(1, 2)}
"""Quantization grids in beats, coarser first."""


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/stats.yaml", type=Path)
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
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
		"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
		"config": config,
		"counts": {"segments": len(segments), "four4": len(four4), "x4": len(x4)},
		"grid": _grid_stats(config, [s.id for s in four4]),
		"four4": _segment_stats(four4),
		"x4": {
			**_segment_stats(x4),
			"barline_crossing_pct": _round(*_barline_crossing(x4)),
		},
	}

	output = Path(config["output_dir"])
	output.mkdir(parents=True, exist_ok=True)
	(output / "stats.json").write_text(
		json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(yaml.safe_dump(_summary(stats), sort_keys=False, allow_unicode=True))


def _segment_stats(segments: list[Segment]) -> dict[str, Any]:
	chord_total = chord_kind = borrowed = applied = 0
	notes_total = notes_over_chord = non_chord = 0
	long_non_chord = resolved = segments_with_long = 0
	for segment in segments:
		melody = segment.melody
		for event in segment.events:
			chord = event.chord
			if chord is None:
				continue
			chord_total += 1
			chord_kind += _has_tension(chord)
			borrowed += chord.borrowed is not None
			applied += chord.applied != 0
		sounding = _chords_at_onsets(melody, segment.events)
		has_long = False
		for index, (note, chord) in enumerate(zip(melody.notes, sounding, strict=True)):
			notes_total += 1
			if chord is None:
				continue
			notes_over_chord += 1
			if note.pitch % 12 in chord_pitch_classes(chord, melody.key_at(note.onset))[1]:
				continue
			non_chord += 1
			if _is_long_on_strong_beat(note, melody):
				long_non_chord += 1
				has_long = True
				if _resolves_stepwise(melody, sounding, index):
					resolved += 1
		segments_with_long += has_long
	return {
		"segments": len(segments),
		"notes": notes_total,
		"non_chord_tone_pct": _round(non_chord, notes_over_chord),
		"long_non_chord_note_pct": _round(long_non_chord, notes_total),
		"long_non_chord_segment_pct": _round(segments_with_long, len(segments)),
		"resolution_pct": _round(resolved, long_non_chord),
		"chord_kind_pct": _round(chord_kind, chord_total),
		"borrowed_pct": _round(borrowed, chord_total),
		"applied_pct": _round(applied, chord_total),
	}


def _has_tension(chord: RawChord) -> bool:
	return (
		chord.type in (9, 11, 13)
		or bool(chord.suspensions)
		or bool(chord.adds)
		or bool(chord.alterations)
	)


def _chords_at_onsets(melody: Melody, events: tuple[ChordEvent, ...]) -> list[RawChord | None]:
	"""Chord sounding at each melody note onset, scanned in a single pass."""
	result: list[RawChord | None] = []
	cursor = 0
	start = 0
	for note in melody.notes:
		while cursor < len(events) and start + events[cursor].duration <= note.onset:
			start += events[cursor].duration
			cursor += 1
		chord = events[cursor].chord if cursor < len(events) and start <= note.onset else None
		result.append(chord)
	return result


def _is_long_on_strong_beat(note: Note, melody: Melody) -> bool:
	if note.onset.denominator != 1:
		return False
	strength = melody.meter.beat_strength(int(note.onset))
	return strength >= 2 and note.duration >= melody.meter.eighths_per_beat


def _resolves_stepwise(melody: Melody, sounding: list[RawChord | None], index: int) -> bool:
	"""Whether the note after a long non-chord tone moves stepwise into a chord tone."""
	if index + 1 >= len(melody.notes):
		return False
	current, following = melody.notes[index], melody.notes[index + 1]
	if abs(following.pitch - current.pitch) > 2:
		return False
	chord = sounding[index + 1]
	return (
		chord is not None
		and following.pitch % 12 in chord_pitch_classes(chord, melody.key_at(following.onset))[1]
	)


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
