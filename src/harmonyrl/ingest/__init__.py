"""Importing melodies from external files (MIDI, MusicXML, MuseScore, PDF, audio)."""

from .common import SheetImport
from .midi import melody_from_midi
from .mscore import melody_from_mscore
from .musicxml import melody_from_musicxml
from .sheet import read_lead_sheet

__all__ = [
	"SheetImport",
	"melody_from_midi",
	"melody_from_mscore",
	"melody_from_musicxml",
	"read_lead_sheet",
]
