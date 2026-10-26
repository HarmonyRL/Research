"""Tonal centroids of Harte, Sandler, and Gasser (2006).

"Detecting harmonic change in musical audio" maps a pitch-class profile to 6 dimensions:
the circles of fifths, minor thirds, and major thirds with radii 1, 1, and 0.5,
divided by the L1 norm of the profile.
"""

from collections.abc import Collection

import numpy as np

_PITCH_CLASSES = np.arange(12)
_BASIS = np.stack(
	[
		np.sin(_PITCH_CLASSES * 7 * np.pi / 6),
		np.cos(_PITCH_CLASSES * 7 * np.pi / 6),
		np.sin(_PITCH_CLASSES * 3 * np.pi / 2),
		np.cos(_PITCH_CLASSES * 3 * np.pi / 2),
		0.5 * np.sin(_PITCH_CLASSES * 2 * np.pi / 3),
		0.5 * np.cos(_PITCH_CLASSES * 2 * np.pi / 3),
	]
)


def tonal_centroid(pitch_classes: Collection[int]) -> np.ndarray:
	"""6-D centroid of the profile with 1 at each of `pitch_classes`, which must not be empty."""
	if not pitch_classes:
		raise ValueError("a tonal centroid needs at least 1 pitch class")

	profile = np.zeros(12)
	profile[list(pitch_classes)] = 1.0
	return _BASIS @ profile / profile.sum()


def tonal_distance(a: Collection[int], b: Collection[int]) -> float:
	"""Euclidean distance between the tonal centroids of 2 pitch-class sets."""
	return float(np.linalg.norm(tonal_centroid(a) - tonal_centroid(b)))
