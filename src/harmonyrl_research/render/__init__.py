"""Rendering melodies with chord events to MIDI, audio, and score formats."""

from .audio import midi_to_wav, wav_to_mp3
from .convert import musicxml_to_formats
from .midi import DEFAULT_TEMPO, write_midi
from .score import DEFAULT_TITLE, melody_to_score, write_score
from .sheet import write_lead_sheet
from .voicing import chord_notes, melody_notes

__all__ = [
	"DEFAULT_TEMPO",
	"DEFAULT_TITLE",
	"chord_notes",
	"melody_notes",
	"melody_to_score",
	"midi_to_wav",
	"musicxml_to_formats",
	"wav_to_mp3",
	"write_lead_sheet",
	"write_midi",
	"write_score",
]
