"""Score format conversion with the MuseScore CLI."""

import subprocess
from collections.abc import Mapping
from pathlib import Path


def musicxml_to_formats(musicxml: Path, targets: Mapping[str, Path], mscore: Path) -> None:
	"""Convert one MusicXML file into several formats, one MuseScore invocation each.

	`targets` maps a MuseScore output extension (pdf, mscz, mscx, png, ...) to its
	destination path. Formats that drop side files next to the output (mscx) must be
	given their own directory by the caller. MuseScore only honors the last `-o` per
	invocation and exits 0 even when it writes nothing, so each target is a separate
	call whose output is verified to exist.
	"""
	for target in targets.values():
		subprocess.run(
			[str(mscore), "-o", str(target), str(musicxml)],
			check=True,
			capture_output=True,
		)
		if not target.exists():
			raise RuntimeError(f"mscore produced no {target} from {musicxml}")
