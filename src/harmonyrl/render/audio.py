"""Audio rendering of MIDI files with FluidSynth."""

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
