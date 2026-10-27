"""Structure critic: the model, its training, and the negatives it learns from."""

from .model import ChordCritic, CriticConfig
from .train import CriticArguments, CriticTrainer

__all__ = ["ChordCritic", "CriticArguments", "CriticConfig", "CriticTrainer"]
