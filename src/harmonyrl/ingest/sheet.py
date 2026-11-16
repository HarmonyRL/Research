"""Reading a lead sheet from any supported file."""

from dataclasses import replace
from pathlib import Path

from harmonyrl.schema import Key, KeySpan, LeadSheet, Meter, SheetPart

from .common import SheetImport, estimate_key
from .midi import lead_sheet_from_midi
from .mscore import lead_sheet_from_mscore
from .musicxml import lead_sheet_from_musicxml


def read_lead_sheet(
	path: Path,
	*,
	part: int | None = None,
	key: Key | None = None,
	meter: Meter | None = None,
	tempo: int | None = None,
	mscore: Path | None = None,
) -> SheetImport:
	"""Read the lead sheet of a score, MIDI, or monophonic recording, chosen by file suffix."""
	suffix = path.suffix.lower()

	if suffix in (".musicxml", ".mxl", ".xml"):
		return lead_sheet_from_musicxml(path, part=part, key=key)
	if suffix in (".mscz", ".mscx"):
		return lead_sheet_from_mscore(path, part=part, key=key, mscore=mscore)
	if suffix in (".mid", ".midi"):
		return lead_sheet_from_midi(path, track=part, key=key)
	if suffix == ".pdf":
		return _from_pdf(path, part=part, key=key, meter=meter)
	if suffix in (".wav", ".mp3"):
		if meter is None or tempo is None:
			raise ValueError("a recording needs meter= and tempo=")
		return _from_audio(path, key=key, meter=meter, tempo=tempo)

	raise ValueError(f"unsupported file type {suffix!r}")


def _from_pdf(path: Path, *, part: int | None, key: Key | None, meter: Meter | None) -> SheetImport:
	# OMR pulls in its own models, so it loads only when a PDF is read.
	from .pdf import melodies_from_pdf

	pages = melodies_from_pdf(path, part=part, meter=meter, key=key)
	notes = tuple(note for page in pages for note in page.melody.notes)
	resolved_key = key or estimate_key(notes)
	parts = tuple(
		SheetPart(replace(page.melody, keys=(KeySpan(0, resolved_key),))) for page in pages
	)
	return SheetImport(LeadSheet(parts), declared_key=pages[0].declared_key if pages else None)


def _from_audio(path: Path, *, key: Key | None, meter: Meter, tempo: int) -> SheetImport:
	# Pitch tracking pulls in librosa, so it loads only when a recording is read.
	from .audio import melody_from_audio

	imported = melody_from_audio(path, meter=meter, tempo=tempo, key=key)
	return SheetImport(LeadSheet((SheetPart(imported.melody),)))
