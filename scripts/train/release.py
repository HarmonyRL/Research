"""Save a trained chord policy as a released model."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import typer

from harmonyrl.models import ChordPolicy, GenerationConfig
from harmonyrl.rl import policy_from_reference

REPO = Path(__file__).resolve().parents[2]


def main(
	source: Annotated[Path, typer.Option()],
	output: Annotated[Path, typer.Option()],
	temperature: Annotated[float, typer.Option()] = 1.0,
) -> None:
	"""Save the policy at `source` into `output` with `temperature` as its default."""
	if output.exists() and any(output.iterdir()):
		raise FileExistsError(f"release directory {output} is not empty")

	policy = policy_from_reference(ChordPolicy.from_pretrained(source))
	policy.save_pretrained(output)
	GenerationConfig(temperature=temperature).save_pretrained(output)

	meta: dict[str, Any] = {
		"git": _git("rev-parse", "--short", "HEAD"),
		"git_dirty": bool(_git("status", "--porcelain")),
		"created": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
		"source": str(source),
		"temperature": temperature,
	}
	(output / "meta.json").write_text(
		json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
	)
	print(f"released {source} into {output}")


def _git(*args: str) -> str:
	return subprocess.run(
		["git", *args], cwd=REPO, check=True, capture_output=True, text=True
	).stdout.strip()


if __name__ == "__main__":
	typer.run(main)
