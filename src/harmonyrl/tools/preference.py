"""Pairwise preference listening sessions on a Streamlit page.

Reads a pairs CSV (pair_id,a_path,b_path,tag), plays the two renditions side by
side with sources hidden and left/right order randomized per session, and appends
one row per choice to the results CSV. Already-answered pairs are skipped, so a
session can be resumed at any time.
"""

import csv
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

_AUDIO_FORMATS = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


@dataclass(frozen=True)
class Pair:
	pair_id: str
	a_path: Path
	b_path: Path
	tag: str


def run_preference(pairs_path: Path, results_path: Path) -> None:
	"""Render the session page; call from a script run by `streamlit run`."""
	pairs = _load_pairs(pairs_path)
	if not pairs:
		st.error(f"no pairs in {pairs_path}")
		return
	if "session" not in st.session_state:
		session = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
		shuffler = random.Random(session)
		st.session_state.session = session
		st.session_state.order = {p.pair_id: shuffler.random() < 0.5 for p in pairs}
		st.session_state.queue = [p for p in pairs if p.pair_id not in _answered_ids(results_path)]
		st.session_state.index = 0
		st.session_state.pair_started = time.monotonic()

	queue = st.session_state.queue
	if st.session_state.index >= len(queue):
		_done_page(results_path)
		return
	_pair_page(queue[st.session_state.index], len(queue), results_path)


def _pair_page(pair: Pair, queue_length: int, results_path: Path) -> None:
	left_is_a = st.session_state.order[pair.pair_id]
	left, right = (pair.a_path, pair.b_path) if left_is_a else (pair.b_path, pair.a_path)
	st.progress(st.session_state.index / max(1, queue_length))
	st.subheader(f"{st.session_state.index + 1} / {queue_length}")

	missing = [path for path in (left, right) if not path.is_file()]
	if missing:
		st.error(f"audio file not found for {pair.pair_id}: {', '.join(str(p) for p in missing)}")
		return

	left_col, right_col = st.columns(2)
	with left_col:
		st.caption("왼쪽 샘플")
		st.audio(left.read_bytes(), format=_audio_format(left))
	with right_col:
		st.caption("오른쪽 샘플")
		st.audio(right.read_bytes(), format=_audio_format(right))

	_show_scores(left, right)

	started = st.session_state.pair_started
	choice = None
	left_btn, tie_btn, right_btn = st.columns(3)
	with left_btn:
		if st.button("◀ 왼쪽 선호", width="stretch"):
			choice = "left"
	with tie_btn:
		if st.button("둘 다 비슷함", width="stretch"):
			choice = "tie"
	with right_btn:
		if st.button("오른쪽 선호 ▶", width="stretch"):
			choice = "right"
	if choice is not None:
		_record(results_path, pair, choice, left_is_a, time.monotonic() - started)
		st.session_state.index += 1
		st.session_state.pair_started = time.monotonic()
		st.rerun()


def _audio_format(path: Path) -> str:
	return _AUDIO_FORMATS.get(path.suffix.lower(), "audio/wav")


def _show_scores(left: Path, right: Path) -> None:
	"""Show both scores full-width in tabs: half-column images are too small to read.

	Each side falls back to its PDF in an iframe when page images were not rendered."""
	if not (_has_score(left) or _has_score(right)):
		return
	left_tab, right_tab = st.tabs(["왼쪽 악보", "오른쪽 악보"])
	with left_tab:
		_show_score(left)
	with right_tab:
		_show_score(right)


def _has_score(audio_path: Path) -> bool:
	return bool(_score_pages(audio_path)) or audio_path.with_suffix(".pdf").is_file()


def _score_pages(audio_path: Path) -> list[Path]:
	"""Page images the rendering pipeline wrote next to the audio, in numeric page order."""
	pattern = re.compile(rf"^{re.escape(audio_path.stem)}-(\d+)\.png$")
	numbered: list[tuple[int, Path]] = []
	for candidate in audio_path.parent.glob(f"{audio_path.stem}-*.png"):
		match = pattern.match(candidate.name)
		if match is not None:
			numbered.append((int(match.group(1)), candidate))
	return [path for _, path in sorted(numbered)]


def _show_score(audio_path: Path) -> None:
	"""Show the score rendered next to the audio, when one exists.

	The rendering pipeline writes each segment's files under one name, so the score
	lives at the audio path with a swapped suffix. Chrome blocks PDFs inside the
	sandboxed iframe, so page images are preferred when the pipeline produced them."""
	pages = _score_pages(audio_path)
	if pages:
		st.image([page.read_bytes() for page in pages])
		return
	score = audio_path.with_suffix(".pdf")
	if score.exists():
		st.iframe(score, height=520)


def _done_page(results_path: Path) -> None:
	rows = _read_results(results_path)
	choices = [row["choice"] for row in rows]
	st.balloons()
	st.header("세션 완료")
	st.metric("응답", len(rows))
	st.metric("왼쪽", choices.count("left"))
	st.metric("비슷함", choices.count("tie"))
	st.metric("오른쪽", choices.count("right"))
	st.caption(f"결과: {results_path}")


def _record(results_path: Path, pair: Pair, choice: str, left_is_a: bool, seconds: float) -> None:
	results_path.parent.mkdir(parents=True, exist_ok=True)
	new = not results_path.exists()
	with results_path.open("a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"session",
				"timestamp",
				"pair_id",
				"choice",
				"left_source",
				"right_source",
				"seconds",
				"tag",
			],
		)
		if new:
			writer.writeheader()
		writer.writerow(
			{
				"session": st.session_state.session,
				"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
				"pair_id": pair.pair_id,
				"choice": choice,
				"left_source": "a" if left_is_a else "b",
				"right_source": "b" if left_is_a else "a",
				"seconds": round(seconds, 1),
				"tag": pair.tag,
			}
		)


def _load_pairs(path: Path) -> list[Pair]:
	if not path.exists():
		return []
	with path.open(encoding="utf-8", newline="") as f:
		return [
			Pair(row["pair_id"], Path(row["a_path"]), Path(row["b_path"]), row.get("tag", ""))
			for row in csv.DictReader(f)
		]


def _read_results(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with path.open(encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def _answered_ids(path: Path) -> set[str]:
	return {row["pair_id"] for row in _read_results(path)}
