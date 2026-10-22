"""Segment serialization to and from JSON lines."""

import json
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from harmonyrl.schema import (
	ChordEvent,
	Key,
	KeySpan,
	Melody,
	Meter,
	Mode,
	Note,
	RawChord,
	Section,
	Segment,
	Split,
)


def segment_to_dict(segment: Segment) -> dict[str, Any]:
	"""JSON-ready nested dict for one Segment. Fractions become "n/d" strings."""
	melody = segment.melody
	return {
		"id": segment.id,
		"split": segment.split.value,
		"melody": {
			"notes": [
				{"onset": str(note.onset), "duration": str(note.duration), "pitch": note.pitch}
				for note in melody.notes
			],
			"meter": {
				"beats_per_bar": melody.meter.beats_per_bar,
				"beat_unit": melody.meter.beat_unit,
			},
			"keys": [
				{"start": span.start, "tonic": span.key.tonic, "mode": span.key.mode.value}
				for span in melody.keys
			],
			"section": melody.section.value if melody.section is not None else None,
			"length": melody.length,
		},
		"events": [
			{
				"chord": None if event.chord is None else _chord_to_dict(event.chord),
				"duration": event.duration,
			}
			for event in segment.events
		],
	}


def segment_from_dict(data: Mapping[str, Any]) -> Segment:
	"""Rebuild a Segment from `segment_to_dict` output."""
	melody = data["melody"]
	return Segment(
		id=data["id"],
		split=Split(data["split"]),
		melody=Melody(
			notes=tuple(
				Note(
					onset=Fraction(note["onset"]),
					duration=Fraction(note["duration"]),
					pitch=note["pitch"],
				)
				for note in melody["notes"]
			),
			meter=Meter(melody["meter"]["beats_per_bar"], melody["meter"]["beat_unit"]),
			keys=tuple(
				KeySpan(start=span["start"], key=Key(tonic=span["tonic"], mode=Mode(span["mode"])))
				for span in melody["keys"]
			),
			section=Section(melody["section"]) if melody["section"] is not None else None,
			length=melody["length"],
		),
		events=tuple(
			ChordEvent(
				chord=None if event["chord"] is None else _chord_from_dict(event["chord"]),
				duration=event["duration"],
			)
			for event in data["events"]
		),
	)


def write_segments(segments: Iterable[Segment], path: Path) -> None:
	"""Write one JSON object per line."""
	with path.open("w", encoding="utf-8") as f:
		for segment in segments:
			f.write(json.dumps(segment_to_dict(segment), ensure_ascii=False) + "\n")


def read_segments(path: Path) -> list[Segment]:
	"""Read back every Segment written by `write_segments`."""
	with path.open(encoding="utf-8") as f:
		return [segment_from_dict(json.loads(line)) for line in f if line.strip()]


def _chord_to_dict(chord: RawChord) -> dict[str, Any]:
	borrowed = chord.borrowed
	if isinstance(borrowed, Mode):
		borrowed = borrowed.value
	elif isinstance(borrowed, tuple):
		borrowed = list(borrowed)
	return {
		"root": chord.root,
		"type": chord.type,
		"inversion": chord.inversion,
		"applied": chord.applied,
		"borrowed": borrowed,
		"adds": list(chord.adds),
		"omits": list(chord.omits),
		"alterations": list(chord.alterations),
		"suspensions": list(chord.suspensions),
	}


def _chord_from_dict(data: Mapping[str, Any]) -> RawChord:
	borrowed = data["borrowed"]
	if isinstance(borrowed, str):
		borrowed = Mode(borrowed)
	elif isinstance(borrowed, list):
		borrowed = tuple(borrowed)
	return RawChord(
		root=data["root"],
		type=data["type"],
		inversion=data["inversion"],
		applied=data["applied"],
		borrowed=borrowed,
		adds=tuple(data["adds"]),
		omits=tuple(data["omits"]),
		alterations=tuple(data["alterations"]),
		suspensions=tuple(data["suspensions"]),
	)
