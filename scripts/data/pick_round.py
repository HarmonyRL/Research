"""Build the round that ranks across policies, for each song of a round of draws, the draw of
each policy ranked 1st in its group there."""

import csv
import json
import random
import string
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import typer

from harmonyrl_research.data import read_segments, write_segments


def main(
	source: Annotated[Path, typer.Option()],
	output: Annotated[Path, typer.Option()],
	seed: Annotated[int, typer.Option()] = 0,
) -> None:
	"""Write to `output` 1 group per song of the round `source`, with the draw ranked 1st of each
	policy there, a random one of them on a tie."""
	if output.exists() and any(output.iterdir()):
		raise FileExistsError(f"round directory {output} is not empty")

	key = json.loads((source / "key.json").read_text(encoding="utf-8"))
	draws = {letter: read_segments(source / name) for letter, name in key["events"].items()}
	ranks: dict[str, dict[str, int]] = {}

	with (source / "results.csv").open(encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			if not row["group_id"].startswith("r"):
				ranks.setdefault(row["group_id"], {})[row["item_id"]] = int(row["rank"])

	rng = random.Random(seed)
	picked: dict[str, dict[int, tuple[str, str]]] = {}

	for group_id, song in key["songs"].items():
		best = min(ranks[group_id].values())
		letter = rng.choice(sorted(item for item, rank in ranks[group_id].items() if rank == best))
		picked.setdefault(song["id"], {})[song["policy"]] = (group_id, letter)

	ids = list(picked)
	rng.shuffle(ids)
	letters = string.ascii_lowercase[: len(key["policies"])]
	output.mkdir(parents=True, exist_ok=True)

	for policy, letter in enumerate(letters):
		write_segments(
			[draws[picked[i][policy][1]][key["songs"][picked[i][policy][0]]["row"]] for i in ids],
			output / f"{letter}.jsonl",
		)

	songs: dict[str, dict[str, Any]] = {}
	with (output / "groups.csv").open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["group_id", "item_id", "path", "tag", "bars"])

		for row, i in enumerate(ids):
			group_id = f"g{row + 1:02d}"
			first = key["songs"][picked[i][0][0]]
			songs[group_id] = {
				"id": i,
				"row": row,
				"shift": first["shift"],
				"seconds": first["seconds"],
				"bars": first["bars"],
				"picks": {
					letter: "".join(picked[i][policy]) for policy, letter in enumerate(letters)
				},
			}

			for policy, letter in enumerate(letters):
				path = source / f"{''.join(picked[i][policy])}.wav"
				writer.writerow([group_id, letter, path.resolve(), "", ceil(first["bars"])])

	(output / "key.json").write_text(
		json.dumps(
			{
				"source": source.name,
				"seed": seed,
				"split": key["split"],
				"k": len(letters),
				"policies": key["policies"],
				"events": {letter: f"{letter}.jsonl" for letter in letters},
				"songs": songs,
				"created": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
			},
			indent=2,
			ensure_ascii=False,
		)
		+ "\n",
		encoding="utf-8",
	)
	print(f"{len(ids)} groups in {output}")


if __name__ == "__main__":
	typer.run(main)
