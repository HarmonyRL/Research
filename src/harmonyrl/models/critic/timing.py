"""Perturbations of when the chord changes: an added change, a missed one, and a moved one.
Each returns the new spans, or None where it cannot apply."""

import numpy as np

from harmonyrl.schema import MAX_DURATION, Segment

from .pool import ChordPool, tones
from .spans import STRONG, TOLERANCE, Span, different, fit, profiles, replace, to_spans


def added_change(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Split a chord at a weak position and give the rest another chord."""
	melody = segment.melody
	spans = to_spans(segment)
	frame_profiles = profiles(melody)

	splits = [
		(i, position)
		for i, (start, end, chord) in enumerate(spans)
		if chord is not None
		for position in range(start + 1, end)
		if melody.meter.beat_strength(position) < STRONG
	]

	for s in rng.permutation(len(splits)).tolist():
		i, position = splits[s]
		start, end, chord = spans[i]
		trial = [*spans[:i], (start, position, chord), (position, end, chord), *spans[i + 1 :]]

		if replace(melody, frame_profiles, trial, i + 1, pool, rng):
			return trial

	return None


def missed_change(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Hold a chord over the next one where that one starts at a strong position."""
	melody = segment.melody
	spans = to_spans(segment)
	frame_profiles = profiles(melody)

	pairs = [
		i
		for i in range(len(spans) - 1)
		if different(melody, spans[i], spans[i + 1])
		and melody.meter.beat_strength(spans[i + 1][0]) >= STRONG
		and spans[i + 1][1] - spans[i][0] <= MAX_DURATION
	]

	for p in rng.permutation(pairs).tolist():
		(start, _, chord), (next_start, end, next_chord) = spans[p], spans[p + 1]
		held = fit(frame_profiles, next_start, end, tones(chord, melody.key_at(start)))
		replaced = fit(
			frame_profiles, next_start, end, tones(next_chord, melody.key_at(next_start))
		)

		if held >= replaced - TOLERANCE:
			return [*spans[:p], (start, end, chord), *spans[p + 2 :]]

	return None


def moved_change(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Move a chord change from a strong position to a weaker one, by an 8th note or a beat.

	The change stays in its key, since a chord is read in the key where it starts.
	"""
	melody = segment.melody
	meter = melody.meter
	spans = to_spans(segment)
	frame_profiles = profiles(melody)

	moves = [
		(i, target)
		for i in range(len(spans) - 1)
		if different(melody, spans[i], spans[i + 1])
		and meter.beat_strength(spans[i + 1][0]) >= STRONG
		for step in (-meter.eighths_per_beat, -1, 1, meter.eighths_per_beat)
		if spans[i][0] < (target := spans[i + 1][0] + step) < spans[i + 1][1]
		and meter.beat_strength(target) < meter.beat_strength(spans[i + 1][0])
		and melody.key_at(target) == melody.key_at(spans[i + 1][0])
	]

	for m in rng.permutation(len(moves)).tolist():
		i, target = moves[m]
		(start, change, chord), (_, end, next_chord) = spans[i], spans[i + 1]

		chord_tones = tones(chord, melody.key_at(start))
		next_tones = tones(next_chord, melody.key_at(change))

		low, high = min(change, target), max(change, target)
		now, then = (chord_tones, next_tones) if target < change else (next_tones, chord_tones)

		if fit(frame_profiles, low, high, then) >= fit(frame_profiles, low, high, now) - TOLERANCE:
			return [*spans[:i], (start, target, chord), (target, end, next_chord), *spans[i + 2 :]]

	return None
