"""Structure metrics: chords that end phrases, chords on repeated phrases, and fit to the key.

These evaluate harmonizations only; no reward uses them.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

from harmonyrl.schema import ChordEvent, Melody

from .phrases import PHRASE_BARS, phrase_ends, phrase_windows
from .spans import ChordSpan, chord_spans


def phrase_end_patterns(
	melody: Melody, events: Sequence[ChordEvent]
) -> tuple[list[str], list[str]]:
	"""Chord pairs that end each phrase, and those that end each bar inside a phrase, as a control.

	A pair is `"<before> <last>"`: the function of the chord before the last one, then of the last
	chord before the boundary. A function is the `root` degree, `applied/root` for an applied chord,
	with `[mode]` added for a borrowed one; `N` is no chord and `^` the start of the segment.
	"""
	spans = chord_spans(melody, events)
	ends = phrase_ends(melody)
	bar = melody.meter.bar_eighths
	inside = [position for position in range(bar, melody.length, bar) if position not in ends]
	return [_pattern(spans, p) for p in ends], [_pattern(spans, p) for p in inside]


def repeated_phrase_agreement(melody: Melody, events: Sequence[ChordEvent]) -> tuple[int, int]:
	"""8th notes where 2 phrases with the same melody carry the same chord, and those compared.

	Phrases are the 4-bar windows of `phrase_windows`. 2 windows have the same melody when their
	notes match in pitch, duration, and onset from the window start. Every pair of matching windows
	with notes is compared on projected chords.
	"""
	spans = chord_spans(melody, events)
	frames = [span.chord for span in spans for _ in range(span.end - span.start)]
	matching: dict[tuple, list[int]] = defaultdict(list)
	for start, end in phrase_windows(melody):
		notes = tuple(
			(note.onset - start, note.duration, note.pitch)
			for note in melody.notes
			if start <= note.onset < end
		)
		if notes:
			matching[notes].append(start)

	size = PHRASE_BARS * melody.meter.bar_eighths
	agreeing = 0
	compared = 0
	for starts in matching.values():
		for a, b in combinations(starts, 2):
			agreeing += sum(frames[a + k] == frames[b + k] for k in range(size))
			compared += size
	return agreeing, compared


def key_consistency(melody: Melody, events: Sequence[ChordEvent]) -> tuple[int, int]:
	"""8th notes whose chord tones all lie in the scale of the current key, and those under a chord.

	A chord held across a key change is judged against the new key after the change.
	"""
	in_key = 0
	under_chord = 0
	for span in chord_spans(melody, events):
		if span.chord is None:
			continue
		for position in range(span.start, span.end):
			key = melody.key_at(position)
			scale = {(key.tonic + step) % 12 for step in key.mode.scale}
			under_chord += 1
			in_key += span.pitch_classes <= scale
	return in_key, under_chord


def _pattern(spans: Sequence[ChordSpan], boundary: int) -> str:
	index = next(i for i, span in enumerate(spans) if span.start < boundary <= span.end)
	before = _function(spans[index - 1]) if index else "^"
	return f"{before} {_function(spans[index])}"


def _function(span: ChordSpan) -> str:
	chord = span.chord
	if chord is None:
		return "N"
	name = f"{chord.applied}/{chord.root}" if chord.applied else str(chord.root)
	return f"{name}[{chord.borrowed}]" if chord.borrowed is not None else name
