"""Audio and score files rendered side by side, shared by the listening tools."""

import re
from pathlib import Path

import streamlit as st

_AUDIO_FORMATS = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


def audio_format(path: Path) -> str:
	"""MIME type of an audio file, falling back to WAV."""
	return _AUDIO_FORMATS.get(path.suffix.lower(), "audio/wav")


def score_pages(audio_path: Path) -> list[Path]:
	"""Page images the rendering pipeline wrote next to the audio, in numeric page order."""
	pattern = re.compile(rf"^{re.escape(audio_path.stem)}-(\d+)\.png$")
	numbered: list[tuple[int, Path]] = []

	for candidate in audio_path.parent.glob(f"{audio_path.stem}-*.png"):
		match = pattern.match(candidate.name)
		if match is not None:
			numbered.append((int(match.group(1)), candidate))

	return [path for _, path in sorted(numbered)]


def has_score(audio_path: Path) -> bool:
	"""Whether a score was rendered next to the audio."""
	return bool(score_pages(audio_path)) or audio_path.with_suffix(".pdf").is_file()


def show_score(audio_path: Path) -> None:
	"""Show the score rendered next to the audio, when one exists."""
	pages = score_pages(audio_path)
	if pages:
		st.image([page.read_bytes() for page in pages])
		return

	score = audio_path.with_suffix(".pdf")
	if score.exists():
		st.iframe(score, height=520)
