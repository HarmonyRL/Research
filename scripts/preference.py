"""Run a pairwise preference listening session.

Usage: streamlit run scripts/preference.py -- --pairs pairs.csv --results results.csv
"""

from pathlib import Path
from typing import Annotated

import typer

from harmonyrl.tools import run_preference


def main(
	pairs: Annotated[Path, typer.Option()],
	results: Annotated[Path, typer.Option()],
) -> None:
	"""Show the pairs of `pairs` and append each choice to `results`."""
	run_preference(pairs, results)


if __name__ == "__main__":
	app = typer.Typer(add_completion=False)
	app.command()(main)
	app(standalone_mode=False)
