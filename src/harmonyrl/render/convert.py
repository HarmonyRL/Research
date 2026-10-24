"""Score format conversion with the MuseScore CLI."""

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

PNG_DPI = 300
"""PNG export resolution. MuseScore's default is ~1200 DPI, which makes page images
of ~140 million pixels: enormous files and past Pillow's decompression-bomb limit."""


def musicxml_to_formats(musicxml: Path, targets: Iterable[Path], mscore: Path) -> None:
	"""Convert one MusicXML file into each target path, one MuseScore invocation each.

	The output format is determined by each target's suffix (.pdf, .mscz, .mscx, ...).
	Formats that drop side files next to the output (mscx) must be given their own
	directory by the caller. MuseScore only honors the last `-o` per invocation and
	exits 0 even when it writes nothing, so each target is a separate call whose
	output is verified to exist. PNG output writes one file per page named
	`<stem>-1.png`, `<stem>-2.png`, ... instead of the requested path, so the check
	accepts either shape.
	"""
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
