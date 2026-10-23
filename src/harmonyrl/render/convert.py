"""Score format conversion with the MuseScore CLI."""

import subprocess
from collections.abc import Iterable
from pathlib import Path


def musicxml_to_formats(musicxml: Path, targets: Iterable[Path], mscore: Path) -> None:
	"""Convert one MusicXML file into each target path, one MuseScore invocation each.

	The output format is determined by each target's suffix (.pdf, .mscz, .mscx, ...).
	Formats that drop side files next to the output (mscx) must be given their own
	directory by the caller. MuseScore only honors the last `-o` per invocation and
	exits 0 even when it writes nothing, so each target is a separate call whose
	output is verified to exist.
	"""
	for target in targets:
		subprocess.run(
			[str(mscore), "-o", str(target), str(musicxml)],
			check=True,
			capture_output=True,
		)
		if not target.exists():
			raise RuntimeError(f"mscore produced no {target} from {musicxml}")
