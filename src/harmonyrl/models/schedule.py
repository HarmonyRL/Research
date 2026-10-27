"""Learning rate schedule shared by the trainers."""

import math
from collections.abc import Callable


def warmup_cosine(warmup: int, total: int) -> Callable[[int], float]:
	"""Learning rate factor: linear warmup, then cosine decay to 0 at step `total`."""

	def factor(step: int) -> float:
		if step < warmup:
			return (step + 1) / warmup
		progress = min(1.0, (step - warmup) / max(1, total - warmup))
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	return factor
