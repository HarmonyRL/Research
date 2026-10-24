"""Importing melodies from external files (MIDI, MusicXML, MuseScore, PDF, audio)."""

from .audio import melody_from_audio
from .common import ImportedMelody
from .midi import melody_from_midi
from .mscore import melody_from_mscore
from .musicxml import melody_from_musicxml
from .pdf import ImportedPageError, melodies_from_pdf

__all__ = [
	"ImportedMelody",
	"ImportedPageError",
	"melodies_from_pdf",
	"melody_from_audio",
	"melody_from_midi",
	"melody_from_mscore",
	"melody_from_musicxml",
]
