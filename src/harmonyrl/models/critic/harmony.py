"""Perturbations of which chord sounds: at a phrase end, in a repeated phrase, between phrases,
and after a key change. Each returns the new spans, or None where it cannot apply.
"""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from harmonyrl.metrics import phrase_ends, phrase_windows
from harmonyrl.schema import Segment, chord_pitch_classes, project

from .pool import ChordPool, tones
from .spans import (
	TOLERANCE,
	Span,
	fit,
	frame_fit,
	from_frames,
	index_at,
	profiles,
	replace,
	to_frames,
	to_spans,
)


def phrase_end(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Replace the chord that ends a phrase."""
	spans = to_spans(segment)
	frame_profiles = profiles(segment.melody)

	for end in rng.permutation(phrase_ends(segment.melody)).tolist():
		i = index_at(spans, end - 1)
		if replace(segment.melody, frame_profiles, spans, i, pool, rng):
			return spans

	return None


def repeat(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Replace the longest chord in 1 of 2 phrases with the same melody."""
	melody = segment.melody
	matching: dict[tuple, list[tuple[int, int]]] = defaultdict(list)

	for start, end in phrase_windows(melody):
		notes = tuple(
			(note.onset - start, note.duration, note.pitch)
			for note in melody.notes
			if start <= note.onset < end
		)

		if notes:
			matching[notes].append((start, end))

	windows = [w for group in matching.values() if len(group) > 1 for w in group]
	spans = to_spans(segment)
	frame_profiles = profiles(melody)

	for w in rng.permutation(len(windows)).tolist():
		start, end = windows[w]
		overlaps = [
			(min(e, end) - max(s, start), i)
			for i, (s, e, chord) in enumerate(spans)
			if chord is not None and s < end and e > start
		]

		if overlaps and replace(melody, frame_profiles, spans, max(overlaps)[1], pool, rng):
			return spans

	return None


def swap(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Exchange the chords of 2 phrases in 1 key, where the melody keeps its chord tones.

	The phrases must differ once their chords are projected, or the exchange would change nothing.
	"""
	melody = segment.melody
	frames, onsets = to_frames(segment)
	labels = [chord and project(chord, melody.key_at(t)) for t, chord in enumerate(frames)]
	frame_profiles = profiles(melody)

	key_of = {
		(start, end): melody.key_at(start)
		for start, end in phrase_windows(melody)
		if all(melody.key_at(t) == melody.key_at(start) for t in range(start, end))
	}

	pairs = [
		(a, b)
		for a, b in combinations(key_of, 2)
		if key_of[a] == key_of[b] and labels[a[0] : a[1]] != labels[b[0] : b[1]]
	]

	for p in rng.permutation(len(pairs)).tolist():
		(a0, a1), (b0, b1) = pairs[p]
		key = key_of[(a0, a1)]

		before = frame_fit(frame_profiles, frames, key, a0, a1, a0) + frame_fit(
			frame_profiles, frames, key, b0, b1, b0
		)
		after = frame_fit(frame_profiles, frames, key, a0, a1, b0) + frame_fit(
			frame_profiles, frames, key, b0, b1, a0
		)

		if after < before - TOLERANCE:
			continue

		new_frames, new_onsets = list(frames), list(onsets)
		new_frames[a0:a1], new_frames[b0:b1] = frames[b0:b1], frames[a0:a1]
		new_onsets[a0:a1], new_onsets[b0:b1] = onsets[b0:b1], onsets[a0:a1]

		return from_frames(new_frames, new_onsets)

	return None


def old_key_chord(segment: Segment, pool: ChordPool, rng: np.random.Generator) -> list[Span] | None:
	"""Make the 1st chord of a new key a chord of the key before, at the same pitches."""
	melody = segment.melody
	spans = to_spans(segment)
	frame_profiles = profiles(melody)

	for k in rng.permutation(range(1, len(melody.keys))).tolist():
		change = melody.keys[k].start
		new_key, old_key = melody.keys[k].key, melody.keys[k - 1].key
		first = next((i for i, (s, _, c) in enumerate(spans) if s == change and c), None)

		if first is None or new_key.mode not in pool.chords:
			continue

		old_start = melody.keys[k - 1].start
		old = Counter(c for s, _, c in spans if old_start <= s < change and c is not None)
		scale = {(new_key.tonic + step) % 12 for step in new_key.mode.scale}

		start, end, original = spans[first]
		baseline = fit(frame_profiles, start, end, tones(original, new_key))

		for chord, _ in old.most_common():
			bass, pitch_classes = chord_pitch_classes(chord, old_key)
			if pitch_classes <= scale:
				continue

			for candidate in pool.chords[new_key.mode]:
				same = chord_pitch_classes(candidate, new_key) == (bass, pitch_classes)
				fits = (
					fit(frame_profiles, start, end, tones(candidate, new_key))
					>= baseline - TOLERANCE
				)

				if same and fits:
					spans[first] = (start, end, candidate)
					return spans

	return None
