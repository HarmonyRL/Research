"""Fit of the reward heads to owner comparisons with a Bradley-Terry loss."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from harmonyrl.schema import Segment

from .batch import collate
from .comparisons import Comparison
from .model import ChordReward

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardArguments:
	"""Settings of `RewardTrainer`, as `transformers.TrainingArguments` for the options used
	here."""

	output_dir: str | Path
	seed: int = 0
	l2: float = 50.0
	steps: int = 200
	folds: int = 10
	curve: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
	batch_size: int = 16


class RewardTrainer:
	"""Fit of the heads of a `ChordReward` over its frozen features, shaped like
	`transformers.Trainer`."""

	def __init__(
		self,
		reward: ChordReward,
		args: RewardArguments,
		comparisons: Sequence[Comparison],
		device: str | None = None,
	) -> None:
		self.reward = reward
		self.args = args
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

		self._songs = np.array([c.song for c in comparisons])
		self._winners = self._embed([c.winner for c in comparisons])
		self._losers = self._embed([c.loser for c in comparisons])

	def train(self) -> dict[str, float]:
		"""Cross-validate, then fit the heads to every comparison and save the reward to
		`output_dir/model`. Returns the corss-validated agreements."""
		metrics = self.cross_validate()
		weight, mean, std = _fit(
			self._winners,
			self._losers,
			self._songs,
			self.args,
			self.reward.config.heads,
			np.random.default_rng(self.args.seed),
		)

		reward = self.reward
		with torch.no_grad():
			reward.heads.weight.copy_(weight)
			reward.heads.bias.zero_()
			reward.feature_mean.copy_(mean)
			reward.feature_std.copy_(std)

		reward.save_pretrained(Path(self.args.output_dir) / "model")
		return metrics

	def cross_validate(self) -> dict[str, float]:
		"""Agreement of the head mean with held-out preference comparisons, songs split into
		`folds`, for each fraction in `curve` of the training songs of a fold."""
		args = self.args
		rng = np.random.default_rng(args.seed)
		songs = rng.permutation(np.unique(self._songs))
		hits: dict[float, list[bool]] = {fraction: [] for fraction in args.curve}

		for test_songs in np.array_split(songs, min(args.folds, len(songs))):
			test = self._mask(np.isin(self._songs, test_songs))
			train_songs = songs[~np.isin(songs, test_songs)]

			for fraction in args.curve:
				kept_songs = train_songs[: max(1, round(fraction * len(train_songs)))]
				kept = self._mask(np.isin(self._songs, kept_songs))
				weight, _, std = _fit(
					self._winners[kept],
					self._losers[kept],
					self._songs[kept.cpu().numpy()],
					args,
					self.reward.config.heads,
					rng,
				)
				margins = ((self._winners[test] - self._losers[test]) / std) @ weight.T
				hits[fraction] += (margins.mean(dim=1) > 0).tolist()

		metrics = {f"agreement_{fraction:g}": float(np.mean(h)) for fraction, h in hits.items()}
		_logger.info(
			"%d comparisons over %d songs, agreement by fraction of training songs: %s",
			len(self._songs),
			len(songs),
			", ".join(f"{k.removeprefix('agreement_')} {v:.3f}" for k, v in metrics.items()),
		)

		return metrics

	def _mask(self, mask: np.ndarray) -> torch.Tensor:
		return torch.from_numpy(mask).to(self.device)

	@torch.no_grad()
	def _embed(self, segments: Sequence[Segment]) -> torch.Tensor:
		"""Encoder features (segments, hidden) of `segments`."""
		reward = self.reward.to(self.device).eval()
		inputs = [reward.token_ids(s.melody, s.events) for s in segments]
		order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]), reverse=True)
		features = torch.zeros(len(inputs), reward.heads.in_features, device=self.device)

		for chunk in batched(order, self.args.batch_size, strict=False):
			batch = collate([inputs[i] for i in chunk]).to(self.device)
			features[list(chunk)] = reward.embed(batch).float()

		return features


def _fit(
	winners: torch.Tensor,
	losers: torch.Tensor,
	songs: np.ndarray,
	args: RewardArguments,
	heads: int,
	rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Head weights (heads, hidden) and the feature mean and std they expect, each head fit by
	L-BFGS to its own bootstrap of the songs."""
	features = torch.cat((winners, losers))
	mean = features.mean(dim=0)
	std = features.std(dim=0).clamp(min=1e-6)
	differences = (winners - losers) / std

	unique, index = np.unique(songs, return_inverse=True)
	draws = rng.integers(len(unique), size=(heads, len(unique)))
	counts = np.stack([np.bincount(draw, minlength=len(unique))[index] for draw in draws])
	weights = torch.tensor(counts, dtype=differences.dtype, device=differences.device)

	weight = torch.zeros(heads, differences.shape[1], device=differences.device, requires_grad=True)
	optimizer = torch.optim.LBFGS([weight], max_iter=args.steps, line_search_fn="strong_wolfe")

	def closure() -> torch.Tensor:
		optimizer.zero_grad()
		margins = differences @ weight.T

		loss = (
			weights.T * functional.softplus(-margins)
		).sum() + args.l2 / 2 * weight.square().sum()
		loss.backward()

		return loss

	optimizer.step(closure)

	return weight.detach(), mean, std
