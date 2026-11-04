"""Run a ranking listening session.

Usage: streamlit run scripts/tools/ranking.py -- --groups groups.csv --results results.csv
"""

from pathlib import Path
from typing import Annotated

import typer

from harmonyrl.tools import run_ranking


def main(
	groups: Annotated[Path, typer.Option()],
	results: Annotated[Path, typer.Option()],
) -> None:
	"""Show the audio groups of `groups` and append each ranking to `results`."""
	run_ranking(groups, results)


if __name__ == "__main__":
	app = typer.Typer(add_completion=False)
	app.command()(main)
	app(standalone_mode=False)
