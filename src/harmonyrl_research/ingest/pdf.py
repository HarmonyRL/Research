"""Import melodies from PDF scores via OMR (homr) on rasterized pages."""

import re
import subprocess
import tempfile
from pathlib import Path

from homr.main import ProcessingConfig, download_weights, process_image
from homr.music_xml_generator import XmlGeneratorArguments

from harmonyrl_research.schema import Key, Meter

from .common import ImportedMelody
from .musicxml import melody_from_musicxml

DPI = 300
"""Rasterization resolution: print standard, what homr's models are trained for."""


def melodies_from_pdf(
	path: Path,
	*,
	part: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
) -> list[ImportedMelody]:
	"""Recognize melodies from every page of a PDF score."""
	with tempfile.TemporaryDirectory() as tmp:
		work = Path(tmp)
		prefix = work / "page"
		subprocess.run(
			["pdftoppm", "-r", str(DPI), "-png", str(path), str(prefix)],
			check=True,
			capture_output=True,
		)
		pages = sorted(work.glob("page-*.png"), key=_page_number)
		if not pages:
			raise RuntimeError(f"pdftoppm produced no pages from {path}")

		config = ProcessingConfig(
			enable_debug=False,
			enable_cache=False,
			write_staff_positions=False,
			read_staff_positions=False,
			selected_staff=-1,
			transformer_use_gpu=True,
			segnet_use_gpu=True,
			coreml_encoder=False,
			title_detection=False,
		)
		# homr's wheel ships without its models; this fetches the missing ones once.
		download_weights(
			segnet_use_gpu=config.segnet_use_gpu,
			transformer_use_gpu=config.transformer_use_gpu,
			coreml_encoder=config.coreml_encoder,
		)

		results: list[ImportedMelody] = []
		failures: list[tuple[int, Exception]] = []

		for index, page in enumerate(pages, start=1):
			try:
				process_image(str(page), config, XmlGeneratorArguments())
				musicxml = page.with_suffix(".musicxml")
				if not musicxml.exists():
					raise RuntimeError(f"homr produced no MusicXML for page {index}")

				results.append(melody_from_musicxml(musicxml, part=part, meter=meter, key=key))
			except Exception as error:
				failures.append((index, error))

	if failures:
		raise ImportedPageError(
			f"OMR failed on page(s) {[index for index, _ in failures]} of {path}",
			results,
			failures,
		)

	return results


def _page_number(path: Path) -> int:
	match = re.search(r"-(\d+)\.png$", path.name)
	return int(match.group(1)) if match else 0


class ImportedPageError(RuntimeError):
	"""OMR failed on some pages. `melodies` holds the pages that succeeded."""

	def __init__(
		self,
		message: str,
		melodies: list[ImportedMelody],
		failures: list[tuple[int, Exception]],
	) -> None:
		super().__init__(message)
		self.melodies = melodies
		self.failures = failures
