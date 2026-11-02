"""Import a melody from MuseScore files (mscz/mscx) via MusicXML conversion."""

import subprocess
import tempfile
from pathlib import Path

from harmonyrl.schema import Key, Meter

from .common import ImportedMelody
from .musicxml import melody_from_musicxml


def melody_from_mscore(
	path: Path,
	*,
	part: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
	mscore: Path | None = None,
) -> ImportedMelody:
	"""Read a melody from an mscz/mscx (or anything MuseScore opens) file.

	The file is converted to MusicXML in a temporary directory and imported
	through the MusicXML path; the original file is only read."""
	resolver = mscore or _default_mscore()
	with tempfile.TemporaryDirectory() as tmp:
		musicxml = Path(tmp) / "converted.musicxml"
		subprocess.run(
			[str(resolver), "-o", str(musicxml), str(path)],
			check=True,
			capture_output=True,
		)
		if not musicxml.exists():
			raise RuntimeError(f"mscore produced no MusicXML from {path}")

		return melody_from_musicxml(musicxml, part=part, meter=meter, key=key)


def _default_mscore() -> Path:
	path = Path.home() / ".local/bin/mscore"
	if not path.exists():
		raise ValueError(
			"mscore not found at ~/.local/bin/mscore; pass mscore=Path(...) explicitly"
		)

	return path
