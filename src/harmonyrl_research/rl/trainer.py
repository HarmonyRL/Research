"""KL-regularized fine-tuning toward the target π_ref · exp(R / β) / Z."""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from harmonyrl_research.models import ChordPolicy, ChordReward
from harmonyrl_research.schema import Segment, merge_repeats

from .local_reward import local_rewards
from .reward import color_rewards, terminal_rewards
from .rollout import Rollout, rollout, rollout_stats

_logger = logging.getLogger(__name__)

_DIVERGENCES = ("reverse", "forward")
_REWARDS = ("color", "learned", "local")


@dataclass(frozen=True)
class RLArguments:
	"""Settings of `RLTrainer`."""

	output_dir: str | Path
	seed: int = 0
	divergence: str = "reverse"
	reward: str = "color"
	beta: float = 0.2
	hold_penalty: float = 2.0
	clip_reward: float = 2.0
	updates: int = 200
	melodies: int = 32
	samples: int = 8
	lr: float = 1.0e-5
	grad_clip: float = 1.0
	eval_every: int = 10
	eval_melodies: int = 64
	save_every: int = 50


def log_ratios(
	log_prob: torch.Tensor, reference_log_prob: torch.Tensor, rewards: torch.Tensor, beta: float
) -> torch.Tensor:
	"""s = R / β + log π_ref(y) - log π(y) of each sample, from log π of its events."""
	return rewards / beta + (reference_log_prob - log_prob).sum(dim=1)


def reverse_weights(s: torch.Tensor, beta: float) -> torch.Tensor:
	"""RLOO weights (melodies, samples): β (s - the mean s of the other samples of the melody)."""
	others = (s.sum(dim=1, keepdim=True) - s) / (s.shape[1] - 1)
	return beta * (s - others)


def forward_weights(s: torch.Tensor) -> torch.Tensor:
	"""DPG weights (melodies, samples): softmax(s) over the samples of each melody, times k."""
	return s.softmax(dim=1) * s.shape[1]


def effective_fraction(s: torch.Tensor) -> torch.Tensor:
	"""Effective sample size of softmax(s) over the samples of each melody, as a fraction of them.

	1 when the target weighs all samples alike, 1 / k when 1 sample takes all the weight.
	"""
	weights = s.softmax(dim=1)
	return 1 / (weights.pow(2).sum(dim=1) * s.shape[1])


class RLTrainer:
	"""Fine-tuning of a policy against a fixed reward, shaped like `transformers.Trainer`."""

	def __init__(
		self,
		policy: ChordPolicy,
		reference: ChordPolicy,
		reward: ChordReward | None,
		args: RLArguments,
		train: Sequence[Segment],
		valid: Sequence[Segment],
		device: str | None = None,
	) -> None:
		if args.samples < 2:
			raise ValueError(f"leave-one-out needs 2 samples or more, got {args.samples}")

		if args.divergence not in _DIVERGENCES:
			raise ValueError(f"divergence must be one of {_DIVERGENCES}, got {args.divergence}")

		if args.reward not in _REWARDS:
			raise ValueError(f"reward must be one of {_REWARDS}, got {args.reward}")
		if args.reward == "learned" and reward is None:
			raise ValueError("the learned reward needs a reward model")
		if args.reward != "learned" and reward is not None:
			raise ValueError(f"the {args.reward} reward takes no reward model")

		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

		self.policy = policy.to(self.device).eval()
		self.reference = reference.to(self.device).eval()
		self.reward = reward.to(self.device).eval() if reward is not None else None

		self.args = args
		self._rng = np.random.default_rng(args.seed)
		self._train = list(train)

		chosen = self._rng.choice(len(valid), min(args.eval_melodies, len(valid)), replace=False)
		self._valid = [valid[i] for i in sorted(chosen.tolist())]

	def train(self) -> dict[str, float]:
		"""Run `args.updates` updates. Returns the metrics of the last one."""
		args = self.args
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		torch.manual_seed(args.seed)

		optimizer = torch.optim.AdamW(
			self.policy.parameters(),
			lr=args.lr,
			weight_decay=0.0,
			fused=self.device.type == "cuda",
		)
		generator = torch.Generator(self.device).manual_seed(args.seed)

		metrics_file = output_dir / "metrics.jsonl"
		metrics_file.write_text("", encoding="utf-8")
		metrics: dict[str, float] = {}

		for update in range(1, args.updates + 1):
			start = time.perf_counter()
			indices = self._rng.choice(len(self._train), args.melodies, replace=False)
			samples = self._rollout([self._train[i] for i in indices.tolist()], generator)

			metrics = {"update": update} | self._metrics("train", samples)
			metrics |= self._step(samples, optimizer)

			if update % args.eval_every == 0 or update == args.updates:
				metrics |= self.evaluate()

			metrics["seconds"] = round(time.perf_counter() - start, 1)

			with metrics_file.open("a", encoding="utf-8") as f:
				f.write(json.dumps(metrics) + "\n")

			_logger.info(
				"update %d reward %.3f kl %.3f objective %.3f unique %.2f (%.1fs)",
				update,
				metrics["train_reward"],
				metrics["train_kl"],
				metrics["train_objective"],
				metrics["train_unique"],
				metrics["seconds"],
			)

			if update % args.save_every == 0:
				self.policy.save_pretrained(output_dir / f"update-{update}")

		self.policy.save_pretrained(output_dir / "final")

		return metrics

	@torch.no_grad()
	def evaluate(self) -> dict[str, float]:
		"""Metrics of the fixed validation segments, sampled with a fixed seed."""
		generator = torch.Generator(self.device).manual_seed(self.args.seed)
		return self._metrics("valid", self._rollout(self._valid, generator))

	def _rollout(self, segments: Sequence[Segment], generator: torch.Generator) -> Rollout:
		"""`args.samples` samples of each segment, those of 1 segment next to each other."""
		repeated = [s for s in segments for _ in range(self.args.samples)]

		return rollout(self.policy, self.reference, self._rewards, repeated, generator=generator)

	def _rewards(self, samples: Sequence[Segment], segments: Sequence[Segment]) -> torch.Tensor:
		if self.args.reward == "color":
			return color_rewards(samples, self.args.hold_penalty)
		if self.args.reward == "local" or self.reward is None:
			return local_rewards(samples, segments)

		return terminal_rewards(
			self.reward,
			[s.melody for s in samples],
			[s.events for s in samples],
			self.args.clip_reward,
		)

	def _log_ratios(self, samples: Rollout) -> torch.Tensor:
		"""s of `samples` as (melodies, samples)."""
		s = log_ratios(
			samples.log_prob, samples.reference_log_prob, samples.rewards, self.args.beta
		)

		return s.view(-1, self.args.samples)

	def _metrics(self, prefix: str, samples: Rollout) -> dict[str, float]:
		k = self.args.samples
		stats = rollout_stats(samples)
		stats["objective"] = stats["reward"] - self.args.beta * stats["kl"]
		stats["effective_fraction"] = effective_fraction(self._log_ratios(samples)).mean().item()
		stats["unique"] = _unique_fraction(samples.samples, k)

		return {f"{prefix}_{name}": value for name, value in stats.items()}

	def _step(self, samples: Rollout, optimizer: torch.optim.Optimizer) -> dict[str, float]:
		s = self._log_ratios(samples)

		if self.args.divergence == "reverse":
			weights = reverse_weights(s, self.args.beta).flatten()
		else:
			weights = forward_weights(s).flatten()

		log_prob = self.policy(samples.batch).log_prob.sum(dim=1)
		loss = -(weights * log_prob).mean()

		optimizer.zero_grad(set_to_none=True)
		loss.backward()

		norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip)
		optimizer.step()

		return {"loss": loss.item(), "grad_norm": norm.item()}


def _unique_fraction(samples: Sequence[Segment], k: int) -> float:
	"""Mean share of distinct harmonizations in each run of k samples, repeats in bars merged."""
	groups = [samples[i : i + k] for i in range(0, len(samples), k)]

	return float(np.mean([len({merge_repeats(s.events, s.melody) for s in g}) / k for g in groups]))
