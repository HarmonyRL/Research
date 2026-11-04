"""Ranking listening sessions on a Streamlit page."""

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from .common import audio_format, has_score, show_score


@dataclass(frozen=True)
class RankItem:
	item_id: str
	path: Path
	tag: str


@dataclass(frozen=True)
class Group:
	group_id: str
	items: tuple[RankItem, ...]


def run_ranking(groups_path: Path, results_path: Path) -> None:
	"""Render the session page; call from a script run by `streamlit run`."""
	groups = _load_groups(groups_path)
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

	_group_page(queue[st.session_state.index], len(queue), results_path)


def _group_page(group: Group, queue_length: int, results_path: Path) -> None:
	ordered = st.session_state.order[group.group_id]

	st.progress(st.session_state.index / max(1, queue_length))
	st.subheader(f"{st.session_state.index + 1} / {queue_length}")

	missing = [item.path for item in ordered if not item.path.is_file()]
	if missing:
		st.error(f"audio file not found for {group.group_id}: {', '.join(str(p) for p in missing)}")
		return

	st.caption("1이 가장 좋습니다. 비슷하면 같은 순위를 고르세요.")
	ranks: dict[str, int | None] = {}
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

	_show_scores(ordered)

	started = st.session_state.group_started
	if st.button("제출", type="primary"):
		unranked = [
			position
			for position, item in enumerate(ordered, start=1)
			if ranks[item.item_id] is None
		]
		if unranked:
			positions = ", ".join(f"{position}번" for position in unranked)
			st.warning(f"모든 음원에 순위를 매겨주세요: {positions}")
		else:
			_record(results_path, group, ordered, ranks, time.monotonic() - started)
			for item in ordered:
				st.session_state.pop(f"rank:{group.group_id}:{item.item_id}", None)
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
	seconds: float,
) -> None:
	results_path.parent.mkdir(parents=True, exist_ok=True)
	new = not results_path.exists()

	with results_path.open("a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"session",
				"timestamp",
				"group_id",
				"item_id",
				"rank",
				"position",
				"seconds",
				"tag",
			],
		)

		if new:
			writer.writeheader()

		for position, item in enumerate(ordered, start=1):
			writer.writerow(
				{
					"session": st.session_state.session,
					"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
					"group_id": group.group_id,
					"item_id": item.item_id,
					"rank": ranks[item.item_id],
					"position": position,
					"seconds": round(seconds, 1),
					"tag": item.tag,
				}
			)


def _load_groups(path: Path) -> list[Group]:
	if not path.exists():
		return []

	grouped: dict[str, list[RankItem]] = {}
	with path.open(encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			grouped.setdefault(row["group_id"], []).append(
				RankItem(row["item_id"], Path(row["path"]), row.get("tag", ""))
			)

	return [Group(group_id, tuple(items)) for group_id, items in grouped.items()]


def _read_results(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []

	with path.open(encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def _answered_ids(path: Path) -> set[str]:
	return {row["group_id"] for row in _read_results(path)}
