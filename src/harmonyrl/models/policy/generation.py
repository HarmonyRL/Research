"""Default generation settings of a chord policy."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

GENERATION_CONFIG_NAME = "generation_config.json"


@dataclass(frozen=True)
class GenerationConfig:
	"""Defaults that the arguments of a generation call override."""

	temperature: float = 1.0
	pool: int = 8

	def __post_init__(self) -> None:
		if self.temperature <= 0:
			raise ValueError(f"temperature must be positive, got {self.temperature}")

		if self.pool < 1:
			raise ValueError(f"pool must be at least 1, got {self.pool}")

	def save_pretrained(self, directory: str | Path) -> None:
		"""Write the settings into `directory` as generation_config.json."""
		directory = Path(directory)
		directory.mkdir(parents=True, exist_ok=True)
		text = json.dumps(asdict(self), indent=2) + "\n"
		(directory / GENERATION_CONFIG_NAME).write_text(text, encoding="utf-8")

	@classmethod
	def from_pretrained(cls, path_or_repo: str | Path, *, revision: str | None = None) -> Self:
		"""Settings saved with a model in a local directory or a Hub repository,
		or the defaults when the model has none."""
		local = Path(path_or_repo)

		if local.is_dir():
			path = local / GENERATION_CONFIG_NAME
			if not path.is_file():
				return cls()
		else:
			try:
				downloaded = hf_hub_download(
					str(path_or_repo), GENERATION_CONFIG_NAME, revision=revision
				)
			except RemoteEntryNotFoundError:
				return cls()

			path = Path(downloaded)

		return cls(**json.loads(path.read_text(encoding="utf-8")))
