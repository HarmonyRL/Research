"""Run a pairwise preference listening session.

Usage: streamlit run scripts/preference.py -- --pairs pairs.csv --results results.csv
"""

import argparse
from pathlib import Path

from harmonyrl.tools import run_preference


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--pairs", type=Path, required=True)
	parser.add_argument("--results", type=Path, required=True)
	args = parser.parse_args()
	run_preference(args.pairs, args.results)


if __name__ == "__main__":
	main()
