"""Import a melody from a MusicXML file."""

from fractions import Fraction
from pathlib import Path

import music21

from harmonyrl.schema import Key, KeySpan, Meter, Mode, Note

from .common import ImportedMelody, build_melody, estimate_key, snap

_BEAT_UNIT = {1: 4, 2: 4, 4: 4, 8: 8, 16: 8}
"""MusicXML beat-type to contract beat units: only x/4 and x/8 are representable."""


def melody_from_musicxml(
	path: Path,
	*,
	part: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
) -> ImportedMelody:
	"""Read a monophonic melody from a MusicXML file.

	`part` selects a part by index; otherwise the part with the most notes wins.
	Rests are dropped, tied notes join into one, chords collapse to their highest
	note, and offsets keep exact fraction values — tuplets survive the import. A
	meter change or an unrepresentable meter raises rather than being dropped."""
	score = music21.converter.parse(str(path))
	voice = _select_part(score, part)
	resolved_meter = meter or _meter_from_score(score)

	notes: list[Note] = []
	for element in voice.recurse().notes:
		# music21 exposes chord symbols through the notes iterator; they are
		# annotation, not melody.
		if isinstance(element, music21.harmony.Harmony):
			continue
		pitch = (
			element.pitch.midi
			if isinstance(element, music21.note.Note)
			else max(p.midi for p in element.pitches)
		)
		onset = snap(Fraction(element.getOffsetInHierarchy(score)) * 2)
		end = onset + Fraction(element.duration.quarterLength) * 2
		if _is_tied_continuation(element) and notes:
			previous = notes[-1]
			notes[-1] = Note(previous.onset, snap(end) - previous.onset, previous.pitch)
			continue
		notes.append(Note(onset, snap(end) - onset, pitch))

	declared = _declared_key(score)
	# A file carries no trace of how long the rests after the last note were, so
	# the melody's length is recovered as the end of the bar containing its last
	# note — the same rule a human would read off the score.
	end = max((note.onset + note.duration for note in notes), default=Fraction(0))
	bar = resolved_meter.bar_eighths
	ceiling = (end + bar - 1) // bar * bar if end else 0
	# A key signature names a major tonic; the relative minor shares it, so the
	# signature only narrows the candidates — the notes pick the mode.
	signature_tonic = None if declared is None else declared.tonic
	resolved_key = (
		key if key is not None else estimate_key(tuple(notes), signature_tonic=signature_tonic)
	)
	melody = build_melody(notes, resolved_meter, (KeySpan(0, resolved_key),), max(1, int(ceiling)))
	return ImportedMelody(melody=melody, declared_key=declared)


def _is_tied_continuation(element) -> bool:
	tie = getattr(element, "tie", None)
	return tie is not None and tie.type in ("continue", "stop")


def _select_part(score, part: int | None):
	parts = list(score.parts) or [score]
	if part is not None:
		if not 0 <= part < len(parts):
			raise ValueError(f"part {part} out of range (0-{len(parts) - 1})")
		return parts[part]
	return max(parts, key=lambda p: len(list(p.recurse().notes)))


def _meter_from_score(score) -> Meter:
	signatures = list(score.recurse().getElementsByClass(music21.meter.TimeSignature))
	if not signatures:
		raise ValueError("no time signature in MusicXML; pass meter=")
	if len(signatures) > 1:
		raise ValueError("time signature changes in MusicXML; split the song and pass meter=")
	ratio = signatures[0]
	if ratio.denominator not in _BEAT_UNIT:
		raise ValueError(f"unsupported beat-type {ratio.denominator}; pass meter=")
	return Meter(ratio.numerator, _BEAT_UNIT[ratio.denominator])


def _declared_key(score) -> Key | None:
	"""Major tonic named by the file's first key signature, if any.

	music21's `asKey()` always names the major key of the signature, so this
	returns a major Key; the mode itself is decided later from the notes."""
	for signature in score.recurse().getElementsByClass(music21.key.KeySignature):
		try:
			named = signature.asKey()
		except music21.exceptions21.Music21Exception:
			return None
		return Key(named.tonic.pitchClass, Mode.MAJOR)
	return None
