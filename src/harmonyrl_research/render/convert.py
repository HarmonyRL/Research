"""Score format conversion with the MuseScore CLI."""

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

PNG_DPI = 300
"""PNG export resolution."""


def musicxml_to_formats(musicxml: Path, targets: Iterable[Path], mscore: Path) -> None:
	"""Convert one MusicXML file into each target path, one MuseScore invocation each."""
	for target in targets:
		command = [str(mscore)]
		if target.suffix.lower() == ".png":
			command += ["--image-resolution", str(PNG_DPI)]
		command += ["-o", str(target), str(musicxml)]

		subprocess.run(command, check=True, capture_output=True)
		if not _produced(target):
			raise RuntimeError(f"mscore produced no {target} from {musicxml}")


def _produced(target: Path) -> bool:
	if target.exists():
		return True

	pattern = re.compile(rf"^{re.escape(target.stem)}-\d+{re.escape(target.suffix)}$")

	return any(
		pattern.match(candidate.name)
		for candidate in target.parent.glob(f"{target.stem}-*{target.suffix}")
	)
