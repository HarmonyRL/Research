"""The 6 metrics of Yeh et al. (2021), "Automatic melody harmonization with triad chords"."""

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise

import numpy as np

from harmonyrl.schema import ChordEvent, Melody

from .spans import ChordSpan, chord_spans, span_at
from .tonal import tonal_distance

_CONSONANCE = {0: 1, 3: 1, 4: 1, 7: 1, 8: 1, 9: 1, 5: 0}
"""Score of the interval from a chord tone up to a melody note; any other interval scores -1."""

_WINDOW = Fraction(1, 2)


def yeh_metrics(melody: Melody, events: Sequence[ChordEvent]) -> dict[str, float | None]:
	"""CHE, CC, CTD, CTnCTR, PCS, and MCTD of `events` harmonizing `melody`.

	A metric is None where it is undefined: CTD without 2 chords in a row,
	the others without a note under a chord.
	"""
	spans = chord_spans(melody, events)
	frames = Counter[object]()
	for span in spans:
		frames[span.chord] += span.end - span.start
	shares = np.array(list(frames.values())) / melody.length

	return {
		"che": float(-(shares * np.log(shares)).sum()),
		"cc": float(len(frames)),
		"ctd": _chord_tonal_distance(spans),
		"ctnctr": _chord_tone_ratio(melody, spans),
		"pcs": _pitch_consonance(melody, spans),
		"mctd": _melody_chord_tonal_distance(melody, spans),
	}


def _chord_tonal_distance(spans: Sequence[ChordSpan]) -> float | None:
	distances = [
		tonal_distance(a.pitch_classes, b.pitch_classes)
		for a, b in pairwise(spans)
		if a.chord is not None and b.chord is not None
	]

	return float(np.mean(distances)) if distances else None


def _chord_tone_ratio(melody: Melody, spans: Sequence[ChordSpan]) -> float | None:
	"""(Chord tones + proper non-chord tones) / notes under a chord.

	A proper non-chord tone is within 2 semitones of the tone after it.
	Each note is judged against the chord at its onset.
	"""
	chord_tones = 0
	non_chord_tones = 0
	proper = 0

	for i, note in enumerate(melody.notes):
		pitch_classes = span_at(spans, note.onset).pitch_classes
		if not pitch_classes:
			continue

		if note.pitch % 12 in pitch_classes:
			chord_tones += 1
		else:
			non_chord_tones += 1
			following = melody.notes[i + 1] if i + 1 < len(melody.notes) else None
			proper += following is not None and abs(following.pitch - note.pitch) <= 2

	notes = chord_tones + non_chord_tones
	return (chord_tones + proper) / notes if notes else None


def _pitch_consonance(melody: Melody, spans: Sequence[ChordSpan]) -> float | None:
	"""Consonance with every chord tone, averaged per 16th-note window a note starts or holds."""
	scores: list[float] = []

	for note in melody.notes:
		window = note.onset - note.onset % _WINDOW

		while window < note.offset:
			pitch_classes = span_at(spans, max(window, note.onset)).pitch_classes

			if window >= note.onset and pitch_classes:
				intervals = [(note.pitch - p) % 12 for p in pitch_classes]
				scores.append(float(np.mean([_CONSONANCE.get(i, -1) for i in intervals])))

			window += _WINDOW

	return float(np.mean(scores)) if scores else None


def _melody_chord_tonal_distance(melody: Melody, spans: Sequence[ChordSpan]) -> float | None:
	"""Tonal distance of each note to the chords under it, weighted by how long they overlap."""
	weighted = 0.0
	weight = Fraction(0)

	for note in melody.notes:
		for span in spans:
			overlap = min(note.offset, span.end) - max(note.onset, span.start)
			if overlap > 0 and span.pitch_classes:
				weighted += float(overlap) * tonal_distance([note.pitch % 12], span.pitch_classes)
				weight += overlap

	return weighted / float(weight) if weight else None
