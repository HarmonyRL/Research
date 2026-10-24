"""Shared helpers for importing melodies from external files."""

from dataclasses import dataclass
from fractions import Fraction

from harmonyrl.schema import Key, KeySpan, Melody, Meter, Mode, Note

SNAP = Fraction(1, 48)
"""Quantization grid in 8th notes: 1/6 of a 16th, so triplets and quintuplets are exact."""


@dataclass(frozen=True)
class ImportedMelody:
	"""An imported melody with the file's tempo for rendering and the key signature
	the file declared (None when absent)."""

	melody: Melody
	tempo: int = 120
	declared_key: Key | None = None


def snap(value: Fraction) -> Fraction:
	"""Round a position to the quantization grid."""
	return Fraction(round(value / SNAP), 48)


def build_melody(
	notes: list[Note],
	meter: Meter,
	keys: tuple[KeySpan, ...],
	length: int,
) -> Melody:
	"""Assemble a contract Melody, dropping rests and merging overlaps.

	Notes must arrive sorted by onset; each overlap keeps the higher pitch, matching
	the monophonic melody the contract requires."""
	cleaned: list[Note] = []
	for note in sorted(notes, key=lambda n: (n.onset, -n.pitch)):
		if note.duration <= 0:
			continue
		if cleaned and note.onset < cleaned[-1].offset:
			if note.pitch <= cleaned[-1].pitch:
				continue
			previous = cleaned[-1]
			cleaned[-1] = Note(previous.onset, note.onset - previous.onset, previous.pitch)
		cleaned.append(note)
	return Melody(notes=tuple(cleaned), meter=meter, keys=keys, section=None, length=length)


def estimate_key(melody_notes: tuple[Note, ...], *, signature_tonic: int | None = None) -> Key:
	"""Pick one key for a melody from its pitch-class distribution.

	With `signature_tonic` (the major tonic a key signature names) only the two
	keys sharing that signature compete: it and its relative minor. Without it,
	every major and minor key competes. External melodies get a single key (no
	modulation detection), so this separates modes, not modulations."""
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
