"""Importing melodies from external files (MIDI, MusicXML, MuseScore, PDF, audio)."""

from .midi import melody_from_midi
from .mscore import melody_from_mscore
from .musicxml import melody_from_musicxml

__all__ = [
	"melody_from_midi",
	"melody_from_mscore",
	"melody_from_musicxml",
]
