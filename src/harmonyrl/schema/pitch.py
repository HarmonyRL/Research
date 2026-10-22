"""Pitch classes of chords, following Hooktheory's chord construction."""

from .chord import RawChord
from .key import Key, Mode

_STACK = (1, 3, 5, 7, 9, 11, 13)
_TONE_COUNT = {5: 3, 7: 4, 9: 5, 11: 6, 13: 7}


def _semitones(scale: tuple[int, ...], index: int) -> int:
	octave, step = divmod(index, 7)
	return scale[step] + 12 * octave


def chord_pitch_classes(chord: RawChord, key: Key) -> tuple[int, frozenset[int]]:
	"""Bass pitch class and the pitch classes of `chord` in `key`.

	Chord tones are stacked in 3rds on the key scale, or on the borrowed scale. An applied chord
	is stacked on the major scale of its target, except that the 7th of vii°7/ is diminished.
	Suspensions and added tones are taken from the same scale. `inversion` counts positions in
	the stack before omissions, so a 2nd inversion keeps the 5th in the bass.
	"""
	if isinstance(chord.borrowed, tuple):
		scale = chord.borrowed
	elif chord.borrowed is not None:
		scale = chord.borrowed.scale
	else:
		scale = key.mode.scale

	root_index = chord.root - 1
	if chord.applied:
		target = scale[root_index]
		scale = tuple(target + step for step in Mode.MAJOR.scale)
		root_index = chord.applied - 1

	root = _semitones(scale, root_index)

	def above_root(degree: int) -> int:
		return _semitones(scale, root_index + degree - 1) - root

	slots = list(_STACK[: _TONE_COUNT[chord.type]])
	if chord.suspensions:
		third = slots.index(3)
		slots[third : third + 1] = chord.suspensions

	tones = {degree: above_root(degree) for degree in slots}
	if chord.applied == 7 and 7 in tones:
		tones[7] -= 1

	for degree in chord.adds:
		tones[degree] = above_root(degree)

	for alteration in chord.alterations:
		degree = int(alteration[1:])
		tones[degree] = above_root(degree) + (1 if alteration[0] == "#" else -1)

	bass = tones[slots[chord.inversion]]
	for degree in chord.omits:
		tones.pop(degree, None)

	pitch_classes = frozenset((key.tonic + root + t) % 12 for t in tones.values())
	return (key.tonic + root + bass) % 12, pitch_classes
