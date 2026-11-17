"""Serve rendered files on localhost for browsing through an SSH tunnel."""

from pathlib import Path
from typing import Annotated

import typer

from harmonyrl_research.tools import serve


def main(
	root: Annotated[Path, typer.Option()] = Path("runs"),
	port: Annotated[int, typer.Option()] = 8000,
) -> None:
	"""Serve the files under `root` on localhost."""
	print(f"serving {root.resolve()} on http://127.0.0.1:{port} (Ctrl+C to stop)")
	serve(root, port)
	print("stopped")


if __name__ == "__main__":
	typer.run(main)
