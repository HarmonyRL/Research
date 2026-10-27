"""Replacement chords for negative examples: the chords of human accompaniments, per mode."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from harmonyrl.schema import Key, Mode, RawChord, Segment, chord_pitch_classes, project


@dataclass(frozen=True, eq=False)
class ChordPool:
	"""Projected chords of human accompaniments per mode, how often each occurs, and its tones.

	`tones[mode]` (chords, 12) marks the pitch classes of each chord in the key of that mode on C.
	"""

	chords: dict[Mode, tuple[RawChord, ...]]
	weights: dict[Mode, np.ndarray]
	tones: dict[Mode, np.ndarray]

	@classmethod
	def from_segments(cls, segments: Sequence[Segment]) -> ChordPool:
		"""The chords of `segments`, which should be training data."""
		counts: dict[Mode, Counter[RawChord]] = defaultdict(Counter)

		for segment in segments:
			start = 0

			for event in segment.events:
				if event.chord is not None:
					key = segment.melody.key_at(start)
					counts[key.mode][project(event.chord, key)] += event.duration

				start += event.duration

		chords = {mode: tuple(c) for mode, c in counts.items()}

		return cls(
			chords=chords,
			weights={
				mode: np.array(list(c.values()), dtype=np.float64) for mode, c in counts.items()
			},
			tones={
				mode: np.stack([tones(c, Key(0, mode)) for c in chords[mode]]) for mode in chords
			},
		)


def tones(chord: RawChord | None, key: Key) -> np.ndarray:
	"""Pitch classes of `chord` in `key`, as 12 zeros and ones; all zeros for no chord."""
	marks = np.zeros(12)

	if chord is not None:
		marks[list(chord_pitch_classes(chord, key)[1])] = 1.0

	return marks
