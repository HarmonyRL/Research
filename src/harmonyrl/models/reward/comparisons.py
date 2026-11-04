"""Preference comparisons: which of 2 harmonizations of 1 melody sounds better."""

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from harmonyrl.data import read_segments
from harmonyrl.schema import Segment

COMPARISONS = "comparisons.jsonl"
"""File of a round directory that lists its comparisons."""


@dataclass(frozen=True)
class EventRef:
	"""Row `row` of the segments file `file` of a round directory."""

	file: str
	row: int


@dataclass(frozen=True)
class ComparisonRef:
	"""The owner preferred `winner` to `loser`, 2 harmonizations of the melody of segment `song`."""

	song: str
	winner: EventRef
	loser: EventRef


@dataclass(frozen=True)
class Comparison:
	"""A comparison with its harmonizations read from the round."""

	song: str
	winner: Segment
	loser: Segment


def write_comparisons(directory: Path, refs: Sequence[ComparisonRef]) -> None:
	"""Write `refs` as the comparisons of the round `directory`."""
	with (directory / COMPARISONS).open("w", encoding="utf-8") as f:
		for ref in refs:
			f.write(json.dumps(asdict(ref)) + "\n")


def read_comparisons(directory: Path) -> list[Comparison]:
	"""The comparisons of the round `directory`, with their harmonizations."""
	files: dict[str, list[Segment]] = {}
	comparisons: list[Comparison] = []

	with (directory / COMPARISONS).open(encoding="utf-8") as f:
		for line in f:
			record = json.loads(line)
			sides: list[Segment] = []

			for side in ("winner", "loser"):
				ref = EventRef(**record[side])
				if ref.file not in files:
					files[ref.file] = read_segments(directory / ref.file)
				sides.append(files[ref.file][ref.row])

			winner, loser = sides
			if winner.id != record["song"] or loser.id != record["song"]:
				raise ValueError(
					f"{directory}: a comparison of {record['song']} points at "
					f"{winner.id}, {loser.id}"
				)

			comparisons.append(Comparison(record["song"], winner, loser))

	return comparisons


def ranking_comparisons(directory: Path) -> list[ComparisonRef]:
	"""Comparisons from the rankings of the round `directory`: every 2 items of a group whose ranks
	differ, the better ranked one winning."""
	key = json.loads((directory / "key.json").read_text(encoding="utf-8"))
	ranks: dict[str, dict[str, int]] = {}

	with (directory / "results.csv").open(encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			if not row["group_id"].startswith("r"):
				ranks.setdefault(row["group_id"], {})[row["item_id"]] = int(row["rank"])

	refs: list[ComparisonRef] = []
	for group, items in ranks.items():
		song = key["songs"][group]

		for a, b in combinations(sorted(items), 2):
			if items[a] == items[b]:
				continue

			winner, loser = (a, b) if items[a] < items[b] else (b, a)
			refs.append(
				ComparisonRef(
					song["id"],
					EventRef(key["events"][winner], song["row"]),
					EventRef(key["events"][loser], song["row"]),
				)
			)

	return refs
