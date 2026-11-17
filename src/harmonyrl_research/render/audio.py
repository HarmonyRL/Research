"""Audio rendering of MIDI files with FluidSynth, and MP3 encoding with ffmpeg."""

import subprocess
from pathlib import Path


def midi_to_wav(
	midi_path: Path,
	wav_path: Path,
	soundfont: Path,
	*,
	sample_rate: int = 44100,
	gain: float = 0.5,
) -> None:
	"""Render one MIDI file to WAV with a General MIDI soundfont."""
	subprocess.run(
		[
			"fluidsynth",
			"-ni",
			"-F",
			str(wav_path),
			"-r",
			str(sample_rate),
			"-g",
			str(gain),
			str(soundfont),
			str(midi_path),
		],
		check=True,
		capture_output=True,
	)


def wav_to_mp3(wav_path: Path, mp3_path: Path, *, bitrate: str = "192k") -> None:
	"""Encode one WAV file to MP3 with libmp3lame."""
	subprocess.run(
		[
			"ffmpeg",
			"-hide_banner",
			"-loglevel",
			"error",
			"-y",
			"-i",
			str(wav_path),
			"-codec:a",
			"libmp3lame",
			"-b:a",
			bitrate,
			str(mp3_path),
		],
		check=True,
		capture_output=True,
	)
