"""Structure critic: negatives for training it, and later the critic itself."""

from .model import ChordCritic, CriticConfig
from .train import CriticArguments, CriticTrainer

__all__ = ["ChordCritic", "CriticArguments", "CriticConfig", "CriticTrainer"]
