"""Chord policy, preference reward, and HMM baseline, with their input features."""

from .batch import Batch, collate
from .features import event_targets, melody_features
from .hmm import ChordHMM, HMMConfig
from .policy import ChordPolicy, PolicyConfig, PolicyTrainer, PretrainArguments
from .reward import (
	ChordReward,
	RewardArguments,
	RewardConfig,
	RewardTrainer,
	best_of_group,
	head_disagreement,
	ranking_comparisons,
	ranking_groups,
	read_comparisons,
	seconds,
	write_comparisons,
)

__all__ = [
	"Batch",
	"ChordHMM",
	"ChordPolicy",
	"ChordReward",
	"HMMConfig",
	"PolicyConfig",
	"PolicyTrainer",
	"PretrainArguments",
	"RewardArguments",
	"RewardConfig",
	"RewardTrainer",
	"best_of_group",
	"collate",
	"event_targets",
	"head_disagreement",
	"melody_features",
	"ranking_comparisons",
	"ranking_groups",
	"read_comparisons",
	"seconds",
	"write_comparisons",
]
