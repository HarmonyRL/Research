"""Rendering melodies with chord events to MIDI, audio, and score formats."""

from .audio import midi_to_wav
from .convert import musicxml_to_formats
from .midi import DEFAULT_TEMPO, write_midi
from .score import melody_to_score, write_score

__all__ = [
	"DEFAULT_TEMPO",
	"melody_to_score",
	"midi_to_wav",
	"musicxml_to_formats",
	"write_midi",
	"write_score",
]
