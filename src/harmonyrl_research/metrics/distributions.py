"""Distances between categorical distributions given as counts."""

import math
from collections.abc import Hashable, Mapping

import numpy as np


def js_distance[K: Hashable](a: Mapping[K, float], b: Mapping[K, float]) -> float:
	"""Jensen-Shannon distance with base-2 logarithms: 0 for equal distributions, 1 for disjoint.

	Counts are normalized, so both must have a positive total.
	"""
	keys = list(set(a) | set(b))
	p = np.array([a.get(key, 0) for key in keys], dtype=np.float64)
	q = np.array([b.get(key, 0) for key in keys], dtype=np.float64)
	if p.sum() <= 0 or q.sum() <= 0:
		raise ValueError("both distributions need a positive total")

	p /= p.sum()
	q /= q.sum()
	m = (p + q) / 2

	def divergence(x: np.ndarray) -> float:
		present = x > 0
		return math.fsum(x[present] * np.log2(x[present] / m[present]))

	return float(np.sqrt(max(0.0, (divergence(p) + divergence(q)) / 2)))
