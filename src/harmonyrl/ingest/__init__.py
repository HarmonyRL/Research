"""Importing melodies from external files (MIDI, MusicXML)."""

from .common import ImportedMelody
from .midi import melody_from_midi
from .musicxml import melody_from_musicxml

__all__ = [
	"ImportedMelody",
	"melody_from_midi",
	"melody_from_musicxml",
]
