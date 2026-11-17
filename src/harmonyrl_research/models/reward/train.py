"""Fit of the reward heads to owner comparisons with a Bradley-Terry loss."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from harmonyrl_research.data import transpose
from harmonyrl_research.schema import Segment

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
	folds: int | None = None
	curve: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
	batch_size: int = 16
	transpositions: Sequence[int] = ()
	pseudo_margin: float | None = None
	pseudo_weight: float = 0.5
	consistency: float = 0.0


class RewardTrainer:
	"""Fit of the heads of a `ChordReward` over its frozen features, shaped like
	`transformers.Trainer`."""

	def __init__(
		self,
		reward: ChordReward,
		args: RewardArguments,
		comparisons: Sequence[Comparison],
		unlabeled: Sequence[tuple[Segment, Segment]] = (),
		device: str | None = None,
	) -> None:
		self.reward = reward
		self.args = args
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

		self._songs = np.array([c.song for c in comparisons])
		self._winners = self._embed([c.winner for c in comparisons])
		self._losers = self._embed([c.loser for c in comparisons])

		self._unlabeled_songs = np.array([a.id for a, _ in unlabeled])
		self._first = self._embed([a for a, _ in unlabeled])
		self._second = self._embed([b for _, b in unlabeled])

	def train(self) -> dict[str, float]:
		"""Cross-validate, then fit the heads to every comparison and save the reward to
		`output_dir/model`. Returns the cross-validated agreements and the pseudo labels used."""
		metrics = self.cross_validate()
		rng = np.random.default_rng(self.args.seed)
		everything = np.ones(len(self._unlabeled_songs), dtype=bool)
		weight, mean, std, pseudo = self._fit_heads(
			self._mask(np.ones(len(self._songs), dtype=bool)), self._mask(everything), rng
		)

		reward = self.reward
		with torch.no_grad():
			reward.heads.weight.copy_(weight)
			reward.heads.bias.zero_()
			reward.feature_mean.copy_(mean)
			reward.feature_std.copy_(std)

		reward.save_pretrained(Path(self.args.output_dir) / "model")
		return metrics | {"pseudo_labels": pseudo}

	def cross_validate(self) -> dict[str, float]:
		"""Agreement of the head mean with held-out preference comparisons in their own key,
		songs split into `folds`, for each fraction in `curve` of the training songs of a fold."""
		args = self.args
		rng = np.random.default_rng(args.seed)
		songs = rng.permutation(np.unique(self._songs))
		hits: dict[float, list[bool]] = {fraction: [] for fraction in args.curve}

		splits = len(songs) if args.folds is None else min(args.folds, len(songs))
		for test_songs in np.array_split(songs, splits):
			test = self._mask(np.isin(self._songs, test_songs))
			unlabeled = self._mask(~np.isin(self._unlabeled_songs, test_songs))
			train_songs = songs[~np.isin(songs, test_songs)]

			for fraction in args.curve:
				kept = train_songs[: max(1, round(fraction * len(train_songs)))]
				weight, _, std, _ = self._fit_heads(
					self._mask(np.isin(self._songs, kept)), unlabeled, rng
				)
				margins = ((self._winners[0, test] - self._losers[0, test]) / std) @ weight.T
				hits[fraction] += (margins[:, 0] > 0).tolist()

		metrics = {f"agreement_{fraction:g}": float(np.mean(h)) for fraction, h in hits.items()}
		_logger.info(
			"%d comparisons over %d songs, %d unlabeled pairs, agreement by fraction of training "
			"songs: %s",
			len(self._songs),
			len(songs),
			len(self._unlabeled_songs),
			", ".join(f"{k.removeprefix('agreement_')} {v:.3f}" for k, v in metrics.items()),
		)

		return metrics

	def _fit_heads(
		self,
		labeled: torch.Tensor,
		unlabeled: torch.Tensor,
		rng: np.random.Generator,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
		"""Heads fit to the `labeled` comparisons, then refit with the pseudo labels of the
		`unlabeled` pairs when those are on.

		Returns the weights, mean, std, and pseudo labels used.
		"""
		args = self.args
		heads = self.reward.config.heads
		winners, losers = self._winners[:, labeled], self._losers[:, labeled]
		songs = self._songs[labeled.cpu().numpy()]
		first, second = self._first[:, unlabeled], self._second[:, unlabeled]
		pool = (first, second) if args.consistency > 0 and first.shape[1] > 0 else None

		weight, mean, std = _fit(winners, losers, songs, args, heads, rng, unlabeled=pool)

		if args.pseudo_margin is None or not bool(unlabeled.any()):
			return weight, mean, std, 0

		pseudo = _pseudo_labels(weight, std, first, second, args.pseudo_margin)
		weight, mean, std = _fit(
			winners, losers, songs, args, heads, rng, pseudo=pseudo, unlabeled=pool
		)

		return weight, mean, std, pseudo[0].shape[1]

	def _mask(self, mask: np.ndarray) -> torch.Tensor:
		return torch.from_numpy(mask).to(self.device)

	def _embed(self, segments: Sequence[Segment]) -> torch.Tensor:
		"""Encoder features (copies, segments, hidden) of `segments` in their own key,
		then transposed by each of `transpositions`."""
		reward = self.reward.to(self.device)
		copies = []

		for shift in (0, *self.args.transpositions):
			moved = [transpose(s, shift) for s in segments]
			copies.append(
				reward.features(
					[s.melody for s in moved], [s.events for s in moved], self.args.batch_size
				)
			)

		return torch.stack(copies)


def _pseudo_labels(
	weight: torch.Tensor,
	std: torch.Tensor,
	first: torch.Tensor,
	second: torch.Tensor,
	margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Unlabeled pairs that every head orders alike in their own key with a mean margin of `margin`
	or more, as the features (copies, pairs, hidden) of their winners and losers."""
	margins = ((first[0] - second[0]) / std) @ weight.T
	unanimous = (margins > 0).all(dim=1) | (margins < 0).all(dim=1)
	sure = unanimous & (margins.mean(dim=1).abs() >= margin)
	forward = (margins.mean(dim=1) > 0)[None, :, None]

	return (
		torch.where(forward, first, second)[:, sure],
		torch.where(forward, second, first)[:, sure],
	)


def _fit(
	winners: torch.Tensor,
	losers: torch.Tensor,
	songs: np.ndarray,
	args: RewardArguments,
	heads: int,
	rng: np.random.Generator,
	pseudo: tuple[torch.Tensor, torch.Tensor] | None = None,
	unlabeled: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Head weights (heads, hidden) and the feature mean and std they expect: the 1st head fit to
	every comparison, the others by L-BFGS to their own bootstrap of the songs."""
	copies = winners.shape[0]
	features = torch.cat((winners, losers), dim=1).flatten(0, 1)
	mean = features.mean(dim=0)
	std = features.std(dim=0).clamp(min=1e-6)
	differences = ((winners - losers) / std).flatten(0, 1)

	unique, index = np.unique(songs, return_inverse=True)
	draws = rng.integers(len(unique), size=(heads - 1, len(unique)))
	counts = np.stack(
		[np.ones(len(songs), dtype=int)]
		+ [np.bincount(draw, minlength=len(unique))[index] for draw in draws]
	)
	weights = torch.tensor(np.tile(counts, copies) / copies, dtype=differences.dtype)
	weights = weights.to(differences.device)

	if pseudo is not None:
		extra = ((pseudo[0] - pseudo[1]) / std).flatten(0, 1)
		differences = torch.cat((differences, extra))
		share = torch.full((heads, len(extra)), args.pseudo_weight / copies, device=weights.device)
		weights = torch.cat((weights, share), dim=1)

	drift = None
	if unlabeled is not None:
		margins = (unlabeled[0] - unlabeled[1]) / std
		drift = (margins[1:] - margins[:1]).flatten(0, 1)

	weight = torch.zeros(heads, differences.shape[1], device=differences.device, requires_grad=True)
	optimizer = torch.optim.LBFGS([weight], max_iter=args.steps, line_search_fn="strong_wolfe")

	def closure() -> torch.Tensor:
		optimizer.zero_grad()
		margins = differences @ weight.T

		loss = (
			weights.T * functional.softplus(-margins)
		).sum() + args.l2 / 2 * weight.square().sum()
		if drift is not None:
			loss = loss + args.consistency * (drift @ weight.T).square().sum()

		loss.backward()
		return loss

	optimizer.step(closure)

	return weight.detach(), mean, std
