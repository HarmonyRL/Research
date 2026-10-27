"""Structure critic: the model and its training."""

from .model import ChordCritic, CriticConfig
from .train import CriticArguments, CriticTrainer

__all__ = ["ChordCritic", "CriticArguments", "CriticConfig", "CriticTrainer"]
