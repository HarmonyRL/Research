"""Shared helpers for importing melodies from external files."""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil

from harmonyrl.schema import Key, KeySpan, LeadSheet, Melody, Meter, Mode, Note

SNAP = Fraction(1, 48)
"""Quantization grid in 8th notes: 1/6 of a 16th, so triplets and quintuplets are exact."""


@dataclass(frozen=True)
class ImportedMelody:
	"""An imported melody with the key signature the file declared (None when absent)."""

	melody: Melody
	declared_key: Key | None = None


@dataclass(frozen=True)
class SheetImport:
	"""An imported lead sheet with the key signature the file declared (None when absent),
	the bars whose chord symbols the vocabulary could only approximate,
	and what else the import did not keep."""

	sheet: LeadSheet
	declared_key: Key | None = None
	approximated: tuple[int, ...] = ()
	warnings: tuple[str, ...] = ()


def snap(value: Fraction) -> Fraction:
	"""Round a position to the quantization grid."""
	return Fraction(round(value / SNAP), 48)


def bar_ceiling(end: Fraction, bar_eighths: int) -> int:
	"""Length in 8th notes covering `end` up to the end of its bar."""
	if end <= 0:
		return 0
	return ceil(end / bar_eighths) * bar_eighths


def meter_from_ratio(numerator: int, denominator: int) -> Meter:
	"""Convert a written time signature into a contract `Meter` with the same bar length.

	The contract only represents `x/4` and `x/8`, so meters like `2/2` or `6/16` are
	rescaled to preserve exact bar length in 8th notes; odd `x/16` bars have half-eighth
	lengths and cannot be represented.
	"""
	if denominator in (1, 2, 4):
		return Meter(numerator * (4 // denominator), 4)
	if denominator == 8:
		return Meter(numerator, 8)
	if denominator == 16 and numerator % 2 == 0:
		return Meter(numerator // 2, 8)

	raise ValueError(f"unsupported time signature {numerator}/{denominator}; pass meter=")


def beat_tempo_from_quarter(quarter_bpm: float, meter: Meter) -> int:
	"""Convert quarter-note BPM into the contract's beat tempo (dotted quarter for x/8)."""
	bpm = quarter_bpm * 2 / 3 if meter.beat_unit == 8 else quarter_bpm
	return max(1, round(bpm))


def eighth_seconds(tempo: int, meter: Meter) -> Fraction:
	"""Duration of one 8th note in seconds for `tempo` (dotted-quarter BPM on x/8)."""
	if tempo < 1:
		raise ValueError(f"tempo must be positive, got {tempo}")
	return Fraction(20, tempo) if meter.beat_unit == 8 else Fraction(30, tempo)


def build_melody(
	notes: list[Note],
	meter: Meter,
	keys: tuple[KeySpan, ...],
	length: int,
	*,
	tempo: int | None,
) -> Melody:
	"""Assemble a contract Melody, dropping rests and merging overlaps."""
	cleaned: list[Note] = []

	for note in sorted(notes, key=lambda n: (n.onset, -n.pitch)):
		if cleaned and note.onset < cleaned[-1].offset:
			if note.pitch <= cleaned[-1].pitch:
				continue

			previous = cleaned[-1]
			cleaned[-1] = Note(previous.onset, note.onset - previous.onset, previous.pitch)

		cleaned.append(note)

	return Melody(
		notes=tuple(cleaned), meter=meter, keys=keys, section=None, tempo=tempo, length=length
	)


def estimate_key(melody_notes: tuple[Note, ...], *, signature_tonic: int | None = None) -> Key:
	"""Pick one key for a melody from its pitch-class distribution."""
	if not melody_notes:
		return Key(signature_tonic if signature_tonic is not None else 0, Mode.MAJOR)

	classes = [note.pitch % 12 for note in melody_notes]
	counts = {pc: classes.count(pc) for pc in range(12)}

	if signature_tonic is None:
		candidates = [(tonic, mode) for tonic in range(12) for mode in (Mode.MAJOR, Mode.MINOR)]
	else:
		candidates = [(signature_tonic, Mode.MAJOR), ((signature_tonic + 9) % 12, Mode.MINOR)]

	best: tuple[int, int] | None = None
	chosen = Key(signature_tonic if signature_tonic is not None else 0, Mode.MAJOR)

	for tonic, mode in candidates:
		candidate = (_scale_score(counts, tonic, mode), -tonic)
		if best is None or candidate > best:
			best = candidate
			chosen = Key(tonic, mode)

	assert best is not None

	return chosen


def _scale_score(counts: dict[int, int], tonic: int, mode: Mode) -> int:
	return sum(count for pc, count in counts.items() if (pc - tonic) % 12 in set(mode.scale))
