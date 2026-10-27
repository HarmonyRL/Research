"""Critic inputs: the melody features of the policy plus the chord sounding at each frame."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from harmonyrl.schema import HEAD_VALUES, ChordEvent, Melody, chord_pitch_classes, project

from ..features import MelodyFeatures, melody_features

CHORD_HEADS = tuple(head for head in HEAD_VALUES if head != "duration")
_INDEX = {head: {value: i for i, value in enumerate(HEAD_VALUES[head])} for head in CHORD_HEADS}


@dataclass(frozen=True)
class ChordFrames:
	"""The chord at each 8th-note frame of a harmonization, in the model vocabulary.

	`values` (frames, heads) indexes `HEAD_VALUES` per chord head; under no chord, `root` is the
	index of None and the other heads take 1 past their last value. `tones` (frames, 12) marks the
	chord's pitch classes from the local tonic and `bass` its bass from the local tonic, 12 under no
	chord. `onset` is True where an event starts.
	"""

	values: np.ndarray
	tones: np.ndarray
	bass: np.ndarray
	onset: np.ndarray


def chord_frames(melody: Melody, events: Sequence[ChordEvent]) -> ChordFrames:
	"""Frames of `events`, which must tile `melody`, with each chord projected in its key."""
	length = melody.length
	values = np.array([len(HEAD_VALUES[head]) for head in CHORD_HEADS], dtype=np.int64)
	values = np.tile(values, (length, 1))
	values[:, 0] = _INDEX["root"][None]
	tones = np.zeros((length, 12), dtype=np.float32)
	bass = np.full(length, 12, dtype=np.int64)
	onset = np.zeros(length, dtype=bool)
	start = 0

	for event in events:
		end = start + event.duration
		onset[start] = True

		if event.chord is not None:
			key = melody.key_at(start)
			chord = project(event.chord, key)
			values[start:end] = [_INDEX[head][getattr(chord, head)] for head in CHORD_HEADS]
			root, pitch_classes = chord_pitch_classes(chord, key)

			for t in range(start, end):
				tonic = melody.key_at(t).tonic
				tones[t, [(p - tonic) % 12 for p in pitch_classes]] = 1.0
				bass[t] = (root - tonic) % 12

		start = end

	if start != length:
		raise ValueError(f"event durations sum to {start}, melody length is {length}")

	return ChordFrames(values=values, tones=tones, bass=bass, onset=onset)


def critic_inputs(
	melody: Melody, events: Sequence[ChordEvent]
) -> tuple[MelodyFeatures, ChordFrames]:
	"""Everything the critic reads about 1 harmonization."""
	return melody_features(melody), chord_frames(melody, events)
