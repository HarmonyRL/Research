"""Ranking listening sessions on a Streamlit page.

When the groups CSV carries a `bars` column, each audio also gets an acceptability
verdict and, for "문제 있음", the bar numbers that are problematic.
"""

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from .common import audio_format, has_score, show_score

_VERDICT_LABELS = {"ok": "수용 가능", "problem": "문제 있음", "unclear": "판단 어려움"}


@dataclass(frozen=True)
class RankItem:
	item_id: str
	path: Path
	tag: str
	bars: int | None


@dataclass(frozen=True)
class Group:
	group_id: str
	items: tuple[RankItem, ...]


def run_ranking(groups_path: Path, results_path: Path) -> None:
	"""Render the session page; call from a script run by `streamlit run`."""
	groups, sectioned = _load_groups(groups_path)
	if not groups:
		st.error(f"no groups in {groups_path}")
		return

	if "session" not in st.session_state:
		session = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")
		shuffler = random.Random(session)
		st.session_state.session = session
		st.session_state.order = {
			group.group_id: tuple(shuffler.sample(group.items, len(group.items)))
			for group in groups
		}
		st.session_state.queue = [
			group for group in groups if group.group_id not in _answered_ids(results_path)
		]
		st.session_state.index = 0
		st.session_state.group_started = time.monotonic()

	queue = st.session_state.queue
	if st.session_state.index >= len(queue):
		_done_page(results_path)
		return

	_group_page(queue[st.session_state.index], len(queue), results_path, sectioned)


def _group_page(group: Group, queue_length: int, results_path: Path, sectioned: bool) -> None:
	ordered = st.session_state.order[group.group_id]

	st.progress(st.session_state.index / max(1, queue_length))
	st.subheader(f"{st.session_state.index + 1} / {queue_length}")

	missing = [item.path for item in ordered if not item.path.is_file()]
	if missing:
		st.error(f"audio file not found for {group.group_id}: {', '.join(str(p) for p in missing)}")
		return

	st.caption("1이 가장 좋습니다. 비슷하면 같은 순위를 고르세요.")
	ranks: dict[str, int | None] = {}
	verdicts: dict[str, str | None] = {}
	problem_bars: dict[str, list[int]] = {}
	for position, item in enumerate(ordered, start=1):
		st.markdown(f"**{position}번**")
		st.audio(item.path.read_bytes(), format=audio_format(item.path))
		ranks[item.item_id] = st.radio(
			"순위",
			options=range(1, len(ordered) + 1),
			index=None,
			horizontal=True,
			key=f"rank:{group.group_id}:{item.item_id}",
		)
		if sectioned:
			verdicts[item.item_id] = st.radio(
				"판정",
				options=list(_VERDICT_LABELS),
				format_func=lambda code: _VERDICT_LABELS[code],
				index=None,
				horizontal=True,
				key=f"verdict:{group.group_id}:{item.item_id}",
			)
			if verdicts[item.item_id] == "problem" and item.bars is not None:
				problem_bars[item.item_id] = st.multiselect(
					"문제 마디",
					options=range(1, item.bars + 1),
					key=f"bars:{group.group_id}:{item.item_id}",
				)

	_show_scores(ordered)

	started = st.session_state.group_started
	if st.button("제출", type="primary"):
		unranked = [
			position
			for position, item in enumerate(ordered, start=1)
			if ranks[item.item_id] is None
		]
		unjudged = [
			position
			for position, item in enumerate(ordered, start=1)
			if sectioned and verdicts.get(item.item_id) is None
		]
		if unranked:
			positions = ", ".join(f"{position}번" for position in unranked)
			st.warning(f"모든 음원에 순위를 매겨주세요: {positions}")
		elif unjudged:
			positions = ", ".join(f"{position}번" for position in unjudged)
			st.warning(f"모든 음원에 판정을 매겨주세요: {positions}")
		else:
			_record(
				results_path,
				group,
				ordered,
				ranks,
				sectioned,
				verdicts,
				problem_bars,
				time.monotonic() - started,
			)
			for item in ordered:
				for prefix in ("rank", "verdict", "bars"):
					st.session_state.pop(f"{prefix}:{group.group_id}:{item.item_id}", None)
			st.session_state.index += 1
			st.session_state.group_started = time.monotonic()
			st.rerun()


def _show_scores(ordered: tuple[RankItem, ...]) -> None:
	"""Show one score tab per position, mirroring the pair tool's layout."""
	if not any(has_score(item.path) for item in ordered):
		return

	tabs = st.tabs([f"{position}번 악보" for position in range(1, len(ordered) + 1)])
	for tab, item in zip(tabs, ordered, strict=True):
		with tab:
			show_score(item.path)


def _done_page(results_path: Path) -> None:
	rows = _read_results(results_path)

	st.balloons()
	st.header("세션 완료")
	st.metric("그룹", len({row["group_id"] for row in rows}))
	st.metric("음원", len(rows))
	st.caption(f"결과: {results_path}")


def _record(
	results_path: Path,
	group: Group,
	ordered: tuple[RankItem, ...],
	ranks: dict[str, int | None],
	sectioned: bool,
	verdicts: dict[str, str | None],
	problem_bars: dict[str, list[int]],
	seconds: float,
) -> None:
	results_path.parent.mkdir(parents=True, exist_ok=True)
	new = not results_path.exists()

	fieldnames = [
		"session",
		"timestamp",
		"group_id",
		"item_id",
		"rank",
		"position",
		"seconds",
		"tag",
	]
	if sectioned:
		fieldnames += ["verdict", "problem_bars"]

	with results_path.open("a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)

		if new:
			writer.writeheader()

		for position, item in enumerate(ordered, start=1):
			row = {
				"session": st.session_state.session,
				"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
				"group_id": group.group_id,
				"item_id": item.item_id,
				"rank": ranks[item.item_id],
				"position": position,
				"seconds": round(seconds, 1),
				"tag": item.tag,
			}
			if sectioned:
				row["verdict"] = verdicts[item.item_id]
				bars = sorted(problem_bars.get(item.item_id, []))
				row["problem_bars"] = ";".join(str(bar) for bar in bars)
			writer.writerow(row)


def _load_groups(path: Path) -> tuple[list[Group], bool]:
	"""Groups and whether the CSV carries the `bars` column for section judging."""
	if not path.exists():
		return [], False

	grouped: dict[str, list[RankItem]] = {}
	with path.open(encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		sectioned = "bars" in (reader.fieldnames or [])
		for row in reader:
			bars = int(row["bars"]) if sectioned and row.get("bars") else None
			grouped.setdefault(row["group_id"], []).append(
				RankItem(row["item_id"], Path(row["path"]), row.get("tag", ""), bars)
			)

	groups = [Group(group_id, tuple(items)) for group_id, items in grouped.items()]
	return groups, sectioned


def _read_results(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []

	with path.open(encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def _answered_ids(path: Path) -> set[str]:
	return {row["group_id"] for row in _read_results(path)}
