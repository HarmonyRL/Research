"""Hugging Face Hub support shared by the trained models."""

import json
from pathlib import Path
from typing import Any, Self

from huggingface_hub import PyTorchModelHubMixin

from harmonyrl.schema import HEAD_VALUES


def head_values_json() -> dict[str, list]:
	"""`HEAD_VALUES` as `config.json` stores it, with modes as strings and tuples as lists."""
	return json.loads(json.dumps(HEAD_VALUES))


class StrictModelHubMixin(PyTorchModelHubMixin):
	"""`PyTorchModelHubMixin` that loads weights strictly by default."""

	@classmethod
	def _from_pretrained(
		cls,
		*,
		model_id: str,
		revision: str | None,
		cache_dir: str | Path | None,
		force_download: bool,
		local_files_only: bool,
		token: str | bool | None,
		map_location: str = "cpu",
		strict: bool = True,
		**model_kwargs: Any,
	) -> Self:
		return super()._from_pretrained(
			model_id=model_id,
			revision=revision,
			cache_dir=cache_dir,
			force_download=force_download,
			local_files_only=local_files_only,
			token=token,
			map_location=map_location,
			strict=strict,
			**model_kwargs,
		)
