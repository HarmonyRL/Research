"""Round-trip check: export segments, import them back, compare melodies."""

import random
from collections import Counter
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Annotated

import typer

from harmonyrl.data import read_segments
from harmonyrl.ingest import melody_from_midi, melody_from_mscore, melody_from_musicxml
from harmonyrl.render import melody_to_score, musicxml_to_formats, write_midi, write_score
from harmonyrl.schema import Melody, Mode

WORKDIR = Path("runs/import-check")
MSCORE = Path.home() / ".local/bin/mscore"


def main(
	sample: Annotated[int, typer.Option()] = 20,
	seed: Annotated[int, typer.Option()] = 20261023,
) -> None:
	"""Round-trip `sample` dataset segments drawn with `seed` through every exported format."""
	segments = [
		segment
		for split in ("train", "valid", "test")
		for segment in read_segments(Path(f"data/processed/segments_{split}.jsonl"))
	]

	# Key signatures encode major and minor only; dataset keys in the other modes
	# (dorian, mixolydian, ...) cannot round-trip through a signature and are out
	# of scope for this check.
	segments = [s for s in segments if s.melody.keys[0].key.mode in (Mode.MAJOR, Mode.MINOR)]
	picked = random.Random(seed).sample(segments, sample)
	WORKDIR.mkdir(parents=True, exist_ok=True)

	results: dict[str, Counter] = {k: Counter() for k in ("midi", "musicxml", "mscz", "mscx")}
	failures: list[str] = []
	mode_misses = Counter()

	for segment in picked:
		base = WORKDIR / segment.id
		write_midi(segment.melody, segment.events, base.with_suffix(".mid"))
		write_score(melody_to_score(segment.melody, segment.events), base.with_suffix(".musicxml"))
		musicxml_to_formats(base.with_suffix(".musicxml"), [base.with_suffix(".mscz")], MSCORE)
		mscx_dir = WORKDIR / f"{segment.id}.mscx.d"
		mscx_dir.mkdir(exist_ok=True)
		musicxml_to_formats(
			base.with_suffix(".musicxml"), [mscx_dir / f"{segment.id}.mscx"], MSCORE
		)

		for kind, imported in (
			("midi", melody_from_midi(base.with_suffix(".mid"))),
			("musicxml", melody_from_musicxml(base.with_suffix(".musicxml"))),
			("mscz", melody_from_mscore(base.with_suffix(".mscz"))),
			("mscx", melody_from_mscore(mscx_dir / f"{segment.id}.mscx")),
		):
			issues, mode_ok = _compare(segment.melody, imported.melody)
			results[kind]["fail" if issues else "ok"] += 1
			if not mode_ok:
				mode_misses[kind] += 1
			if issues:
				failures.append(f"{segment.id} [{kind}]: {issues}")

	for kind, counter in results.items():
		print(f"{kind}: ok {counter['ok']}/{counter['ok'] + counter['fail']}")

	print(
		f"key mode estimated wrong (parallel-key): midi {mode_misses['midi']}, "
		f"musicxml {mode_misses['musicxml']}, mscz {mode_misses['mscz']}, "
		f"mscx {mode_misses['mscx']} — melody notes alone cannot always "
		f"decide major vs relative minor"
	)

	for line in failures[:20]:
		print(f"  {line}")


def _compare(original: Melody, imported: Melody) -> tuple[list[str], bool]:
	"""Hard issues (notes, meter, tempo, length) and whether the key mode landed on the
	expected key rather than its parallel partner."""
	issues: list[str] = []

	if imported.meter != original.meter:
		issues.append(f"meter {imported.meter} != {original.meter}")

	if original.tempo is not None and imported.tempo != original.tempo:
		issues.append(f"tempo {imported.tempo} != {original.tempo}")

	# The importer recovers length as the end of the last note's bar, so the
	# expected value is the original normalized to the same rule.
	bar = original.meter.bar_eighths
	end = max((note.offset for note in original.notes), default=Fraction(0))
	expected_length = max(1, ceil(end / bar) * bar)

	if imported.length != expected_length:
		issues.append(f"length {imported.length} != {expected_length}")

	expected_key = original.keys[0].key

	if imported.keys[0].key != expected_key:
		relative = (
			(expected_key.tonic + 9) % 12
			if expected_key.mode == Mode.MAJOR
			else (expected_key.tonic - 9) % 12
		)

		if imported.keys[0].key.tonic not in (expected_key.tonic, relative):
			issues.append(f"key {imported.keys[0].key} shares no signature with {expected_key}")

	if len(imported.notes) != len(original.notes):
		issues.append(f"note count {len(imported.notes)} != {len(original.notes)}")
		return issues, False

	for mine, theirs in zip(original.notes, imported.notes, strict=True):
		if (mine.onset, mine.duration, mine.pitch) != (
			theirs.onset,
			theirs.duration,
			theirs.pitch,
		):
			issues.append(f"note {mine} != {theirs}")
			break

	return issues, imported.keys[0].key == expected_key


if __name__ == "__main__":
	typer.run(main)
