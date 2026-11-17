"""Loading the two source files and checking timing alignment between them."""

import gzip
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any


def load_json_gz(path: Path) -> dict[str, Any]:
	"""Parse a gzipped JSON file of items keyed by Hooktheory ID."""
	with gzip.open(path, "rt", encoding="utf-8") as f:
		return json.load(f)


def beat_value(value: float | int) -> Fraction:
	"""Exact beat position from a source number, absorbing float micro-errors."""
	if isinstance(value, float):
		return Fraction(value).limit_denominator(48)
	return Fraction(value)


def _non_rest_spans(raw_item: Mapping[str, Any]) -> list[tuple[Fraction, Fraction]]:
	spans = []
	for chord in raw_item["json"].get("chords") or ():
		if chord.get("isRest"):
			continue
		onset = beat_value(chord["beat"]) - 1
		spans.append((onset, onset + beat_value(chord["duration"])))
	return spans


def _harmony_spans(simple_item: Mapping[str, Any]) -> list[tuple[Fraction, Fraction]]:
	return [
		(beat_value(h["onset"]), beat_value(h["offset"]))
		for h in simple_item["annotations"]["harmony"] or ()
	]


@dataclass(frozen=True)
class AlignmentReport:
	"""Per-item equality of raw non-rest chord boundaries and simplified harmony boundaries."""

	compared: int
	equal: int
	differing: list[str] = field(default_factory=list)


def alignment_report(
	simple: Mapping[str, Mapping[str, Any]],
	raw: Mapping[str, Mapping[str, Any]],
	ids: Iterable[str],
) -> AlignmentReport:
	"""Compare boundary sequences of every item in `ids`, collecting the differing IDs."""
	equal = 0
	differing: list[str] = []
	for sid in ids:
		if _non_rest_spans(raw[sid]) == _harmony_spans(simple[sid]):
			equal += 1
		else:
			differing.append(sid)
	return AlignmentReport(compared=equal + len(differing), equal=equal, differing=differing)
