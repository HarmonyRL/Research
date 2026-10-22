"""Filtering joined items and converting them into contract Segments."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor
from typing import Any

from harmonyrl.schema import (
	MAX_DURATION,
	ChordEvent,
	Key,
	KeySpan,
	Melody,
	Meter,
	Mode,
	Note,
	RawChord,
	Segment,
	Split,
	mode_from_intervals,
	section_from_label,
)

from .join import beat_value


@dataclass
class PreparationReport:
	"""Counts of items at each filtering stage and of every deviation from the sources."""

	total: int = 0
	both_present: int = 0
	meter_change: int = 0
	used: int = 0
	with_modulation: int = 0
	segments: int = 0
	items_without_keys: list[str] = field(default_factory=list)
	length_trimmed: list[str] = field(default_factory=list)
	dropped_no_chord_overlong: list[str] = field(default_factory=list)
	root_zero_as_no_chord: int = 0
	borrowed_unknown: list[str] = field(default_factory=list)
	overlaps_clipped: int = 0
	chords_overlapped_away: int = 0
	chords_lost_quantized: int = 0
	no_chord_collapsed: int = 0
	key_starts_snapped: int = 0
	key_spans_dropped_at_end: int = 0


def prepare_dataset(
	simple: Mapping[str, Mapping[str, Any]],
	raw: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Segment], PreparationReport]:
	"""Convert every usable single-meter item shared by both sources into a Segment."""
	report = PreparationReport(total=len(simple))
	ids, both_present, meter_change = usable_ids(simple, raw)
	report.both_present = both_present
	report.meter_change = meter_change
	report.used = len(ids)
	segments: list[Segment] = []
	for sid in ids:
		simple_item = simple[sid]
		if len(simple_item["annotations"]["keys"]) > 1:
			report.with_modulation += 1
		segment = _convert_item(sid, simple_item, raw[sid], report)
		if segment is not None:
			segments.append(segment)
	report.segments = len(segments)
	return segments, report


def usable_ids(
	simple: Mapping[str, Mapping[str, Any]],
	raw: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], int, int]:
	"""IDs with melody and harmony in both sources and a single meter, plus stage counts.

	The counts are items with both melody and harmony, and how many of those were then
	excluded for a meter change inside the item.
	"""
	ids: list[str] = []
	both_present = 0
	meter_change = 0
	for sid, simple_item in simple.items():
		if sid not in raw or not _usable(simple_item):
			continue
		both_present += 1
		if len(simple_item["annotations"]["meters"]) > 1:
			meter_change += 1
			continue
		ids.append(sid)
	return ids, both_present, meter_change


def tempos_by_id(raw: Mapping[str, Mapping[str, Any]], ids: Iterable[str]) -> dict[str, int]:
	"""First recorded tempo in bpm per item, skipping items without a usable value.

	Rendering plays each segment at its source tempo; items with no tempo (or 0) fall
	back to a fixed default at render time.
	"""
	tempos: dict[str, int] = {}
	for sid in ids:
		entries = (raw[sid].get("json") or {}).get("tempos") or []
		bpm = entries[0].get("bpm") if entries else None
		if isinstance(bpm, int) and bpm > 0:
			tempos[sid] = bpm
	return tempos


def _usable(simple_item: Mapping[str, Any]) -> bool:
	annotations = simple_item["annotations"]
	return bool(annotations["melody"]) and bool(annotations["harmony"])


def _snap(value: Fraction) -> int:
	return floor(value + Fraction(1, 2))


def _raw_chord(chord: Mapping[str, Any], report: PreparationReport) -> RawChord | None:
	"""Contract chord for one raw chord entry, or None when no chord sounds."""
	if chord.get("isRest"):
		return None
	if chord["root"] == 0:
		report.root_zero_as_no_chord += 1
		return None
	borrowed = chord.get("borrowed")
	if isinstance(borrowed, str):
		if borrowed == "":
			borrowed = None
		else:
			try:
				borrowed = Mode(borrowed)
			except ValueError:
				report.borrowed_unknown.append(borrowed)
				borrowed = None
	elif isinstance(borrowed, list):
		borrowed = tuple(borrowed)
	return RawChord(
		root=chord["root"],
		type=chord["type"],
		inversion=chord["inversion"],
		applied=chord["applied"],
		borrowed=borrowed,
		adds=tuple(chord.get("adds") or ()),
		omits=tuple(chord.get("omits") or ()),
		alterations=tuple(chord.get("alterations") or ()),
		suspensions=tuple(chord.get("suspensions") or ()),
	)


def _chord_spans(
	chords: list[Mapping[str, Any]],
	scale: Fraction,
	report: PreparationReport,
) -> list[tuple[Fraction, Fraction, RawChord | None]]:
	"""Chord spans in 8th notes, clipped where annotations overlap."""
	spans = []
	for chord in chords:
		onset = (beat_value(chord["beat"]) - 1) * scale
		end = onset + beat_value(chord["duration"]) * scale
		spans.append((onset, end, _raw_chord(chord, report)))
	spans.sort(key=lambda span: span[0])
	clipped: list[tuple[Fraction, Fraction, RawChord | None]] = []
	for i, span in enumerate(spans):
		onset, end, chord = span
		if i + 1 < len(spans) and end > spans[i + 1][0]:
			report.overlaps_clipped += 1
			end = spans[i + 1][0]
		if end <= onset:
			report.chords_overlapped_away += 1
			continue
		clipped.append((onset, end, chord))
	return clipped


def _events(
	spans: list[tuple[Fraction, Fraction, RawChord | None]],
	length: int,
	report: PreparationReport,
) -> list[ChordEvent] | None:
	"""Events tiling [0, length] on the 8th-note grid, or None when a no-chord stretch
	longer than MAX_DURATION cannot be represented as a single event."""
	events: list[tuple[RawChord | None, int]] = []
	cursor = 0
	for onset, end, chord in spans:
		start_i, end_i = _snap(onset), _snap(end)
		if end_i <= start_i:
			if chord is None:
				report.no_chord_collapsed += 1
			else:
				report.chords_lost_quantized += 1
			continue
		if start_i > cursor:
			events.append((None, start_i - cursor))
		events.append((chord, end_i - start_i))
		cursor = end_i
	if length > cursor:
		events.append((None, length - cursor))

	merged: list[tuple[RawChord | None, int]] = []
	for chord, duration in events:
		if merged and merged[-1][0] is None and chord is None:
			merged[-1] = (None, merged[-1][1] + duration)
		else:
			merged.append((chord, duration))

	if any(chord is None and duration > MAX_DURATION for chord, duration in merged):
		return None
	return [ChordEvent(chord, duration) for chord, duration in merged]


def _key_spans(
	simple_item: Mapping[str, Any],
	meter: Meter,
	length: int,
	report: PreparationReport,
) -> tuple[KeySpan, ...]:
	"""Local keys over [0, length], dropping only spans that start at or beyond the end."""
	scale = Fraction(8, meter.beat_unit)
	spans = []
	for key in simple_item["annotations"]["keys"]:
		exact = beat_value(key["beat"]) * scale
		start = _snap(exact)
		report.key_starts_snapped += start != exact
		mode = mode_from_intervals(key["scale_degree_intervals"])
		spans.append((start, Key(key["tonic_pitch_class"], mode)))

	deduped = [spans[0]]
	for start, key in spans[1:]:
		if start == deduped[-1][0]:
			deduped[-1] = (start, key)
		else:
			deduped.append((start, key))
	within = [(start, key) for start, key in deduped if start < length]
	report.key_spans_dropped_at_end += len(deduped) - len(within)

	distinct = [within[0]]
	for span in within[1:]:
		if span[1] != distinct[-1][1]:
			distinct.append(span)
	return tuple(KeySpan(start, key) for start, key in distinct)


def _convert_item(
	sid: str,
	simple_item: Mapping[str, Any],
	raw_item: Mapping[str, Any],
	report: PreparationReport,
) -> Segment | None:
	annotations = simple_item["annotations"]
	if not annotations["keys"]:
		report.items_without_keys.append(sid)
		return None
	meter_entry = annotations["meters"][0]
	meter = Meter(meter_entry["beats_per_bar"], meter_entry["beat_unit"])
	scale = Fraction(8, meter.beat_unit)

	notes = tuple(
		Note(
			onset=beat_value(note["onset"]) * scale,
			duration=(beat_value(note["offset"]) - beat_value(note["onset"])) * scale,
			pitch=60 + 12 * note["octave"] + note["pitch_class"],
		)
		for note in annotations["melody"]
	)
	chords = _chord_spans(raw_item["json"].get("chords") or [], scale, report)

	content_end = notes[-1].offset if notes else Fraction(0)
	if chords:
		content_end = max(content_end, chords[-1][1])
	length = ceil(beat_value(annotations["num_beats"]) * scale)
	if length - content_end > MAX_DURATION:
		length = ceil(content_end)
		report.length_trimmed.append(sid)

	events = _events(chords, length, report)
	if events is None:
		report.dropped_no_chord_overlong.append(sid)
		return None

	melody = Melody(
		notes=notes,
		meter=meter,
		keys=_key_spans(simple_item, meter, length, report),
		section=section_from_label(raw_item["json_api"].get("section") or ""),
		length=length,
	)
	return Segment(
		id=sid,
		split=Split(simple_item["split"].lower()),
		melody=melody,
		events=tuple(events),
	)
