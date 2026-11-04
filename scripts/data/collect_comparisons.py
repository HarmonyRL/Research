"""Write the comparisons of a ranking round from the owner's rankings.

Usage: python scripts/data/collect_comparisons.py --round data/preferences/round-1
"""

from pathlib import Path
from typing import Annotated

import typer

from harmonyrl.models import ranking_comparisons, write_comparisons


def main(round_dir: Annotated[Path, typer.Option("--round")]) -> None:
	"""Collect the comparisons of `round_dir` into its comparisons file."""
	refs = ranking_comparisons(round_dir)
	write_comparisons(round_dir, refs)
	print(f"{len(refs)} comparisons over {len({r.song for r in refs})} songs")


if __name__ == "__main__":
	typer.run(main)
