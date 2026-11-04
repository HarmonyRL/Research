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
	pseudo_margin: float | None = None
	pseudo_weight: float = 0.5


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
		"""Agreement of the head mean with held-out preference comparisons, songs split into
		`folds`, for each fraction in `curve` of the training songs of a fold.

		With pseudo labels on, the same without them is reported as `baseline_agreement_*`.
		"""
		args = self.args
		rng = np.random.default_rng(args.seed)
		songs = rng.permutation(np.unique(self._songs))
		pseudo = args.pseudo_margin is not None and len(self._unlabeled_songs) > 0
		hits: dict[str, list[bool]] = {}

		for test_songs in np.array_split(songs, min(args.folds, len(songs))):
			test = self._mask(np.isin(self._songs, test_songs))
			unlabeled = self._mask(~np.isin(self._unlabeled_songs, test_songs))
			train_songs = songs[~np.isin(songs, test_songs)]

			for fraction in args.curve:
				kept = train_songs[: max(1, round(fraction * len(train_songs)))]
				labeled = self._mask(np.isin(self._songs, kept))
				runs = {"agreement": unlabeled}
				if pseudo:
					runs["baseline_agreement"] = self._mask(
						np.zeros(len(self._unlabeled_songs), dtype=bool)
					)

				for name, pool in runs.items():
					weight, _, std, _ = self._fit_heads(labeled, pool, rng)
					margins = ((self._winners[test] - self._losers[test]) / std) @ weight.T
					hits.setdefault(f"{name}_{fraction:g}", []).extend(
						(margins.mean(dim=1) > 0).tolist()
					)

		metrics = {name: float(np.mean(h)) for name, h in hits.items()}
		_logger.info(
			"%d comparisons over %d songs, %d unlabeled pairs: %s",
			len(self._songs),
			len(songs),
			len(self._unlabeled_songs),
			", ".join(f"{k} {v:.3f}" for k, v in metrics.items()),
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
		winners, losers = self._winners[labeled], self._losers[labeled]
		songs = self._songs[labeled.cpu().numpy()]
		weight, mean, std = _fit(winners, losers, songs, args, self.reward.config.heads, rng)

		if args.pseudo_margin is None or not bool(unlabeled.any()):
			return weight, mean, std, 0

		extra = _pseudo_labels(
			weight, std, self._first[unlabeled], self._second[unlabeled], args.pseudo_margin
		)
		weight, mean, std = _fit(
			winners, losers, songs, args, self.reward.config.heads, rng, extra=extra
		)

		return weight, mean, std, len(extra[0])

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


def _pseudo_labels(
	weight: torch.Tensor,
	std: torch.Tensor,
	first: torch.Tensor,
	second: torch.Tensor,
	margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Unlabeled pairs that every head orders alike with a mean margin of `margin` or more,
	as the features of their winners and losers."""
	margins = ((first - second) / std) @ weight.T
	unanimous = (margins > 0).all(dim=1) | (margins < 0).all(dim=1)
	sure = unanimous & (margins.mean(dim=1).abs() >= margin)
	forward = (margins.mean(dim=1) > 0)[:, None]

	return torch.where(forward, first, second)[sure], torch.where(forward, second, first)[sure]


def _fit(
	winners: torch.Tensor,
	losers: torch.Tensor,
	songs: np.ndarray,
	args: RewardArguments,
	heads: int,
	rng: np.random.Generator,
	extra: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Head weights (heads, hidden) and the feature mean and std they expect, each head fit by
	L-BFGS to its own bootstrap of the songs. `extra` comparisons, the pseudo labels, enter every
	head with weight `args.pseudo_weight`."""
	features = torch.cat((winners, losers))
	mean = features.mean(dim=0)
	std = features.std(dim=0).clamp(min=1e-6)
	differences = (winners - losers) / std

	unique, index = np.unique(songs, return_inverse=True)
	draws = rng.integers(len(unique), size=(heads, len(unique)))
	counts = np.stack([np.bincount(draw, minlength=len(unique))[index] for draw in draws])
	weights = torch.tensor(counts, dtype=differences.dtype, device=differences.device)

	if extra is not None:
		differences = torch.cat((differences, (extra[0] - extra[1]) / std))
		pseudo = torch.full((heads, len(extra[0])), args.pseudo_weight, device=weights.device)
		weights = torch.cat((weights, pseudo), dim=1)

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
