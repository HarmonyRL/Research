"""Import a melody from MuseScore files (mscz/mscx) via MusicXML conversion."""

import subprocess
import tempfile
from pathlib import Path

from harmonyrl.schema import Key, Meter

from .common import ImportedMelody, SheetImport
from .musicxml import lead_sheet_from_musicxml, melody_from_musicxml


def melody_from_mscore(
	path: Path,
	*,
	part: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
	mscore: Path | None = None,
) -> ImportedMelody:
	"""Read a melody from an mscz/mscx (or anything MuseScore opens) file."""
	with tempfile.TemporaryDirectory() as tmp:
		musicxml = _to_musicxml(path, Path(tmp), mscore)
		return melody_from_musicxml(musicxml, part=part, meter=meter, key=key)


def lead_sheet_from_mscore(
	path: Path, *, part: int | None = None, key: Key | None = None, mscore: Path | None = None
) -> SheetImport:
	"""Read a lead sheet from an mscz/mscx file through its MusicXML conversion."""
	with tempfile.TemporaryDirectory() as tmp:
		return lead_sheet_from_musicxml(_to_musicxml(path, Path(tmp), mscore), part=part, key=key)


def _to_musicxml(path: Path, directory: Path, mscore: Path | None) -> Path:
	musicxml = directory / "converted.musicxml"
	subprocess.run(
		[str(mscore or _default_mscore()), "-o", str(musicxml), str(path)],
		check=True,
		capture_output=True,
	)
	if not musicxml.exists():
		raise RuntimeError(f"mscore produced no MusicXML from {path}")

	return musicxml


def _default_mscore() -> Path:
	path = Path.home() / ".local/bin/mscore"
	if not path.exists():
		raise ValueError(
			"mscore not found at ~/.local/bin/mscore; pass mscore=Path(...) explicitly"
		)

	return path
