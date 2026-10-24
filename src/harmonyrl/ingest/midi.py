"""Import a melody from a MIDI file."""

from fractions import Fraction
from pathlib import Path

import pretty_midi

from harmonyrl.schema import Key, KeySpan, Meter, Note

from .common import ImportedMelody, build_melody, estimate_key, snap

DEFAULT_TEMPO = 120
"""Quarter-note tempo assumed when the file carries none."""


def melody_from_midi(
	path: Path,
	*,
	track: int | None = None,
	meter: Meter | None = None,
	key: Key | None = None,
) -> ImportedMelody:
	"""Read a monophonic melody from a MIDI file.

	`track` selects an instrument by index; otherwise the track with the most
	notes in the upper register wins. Chords collapse to their highest note and
	positions snap to 1/48 of an 8th note, which keeps triplets exact. A meter
	change or a missing meter raises rather than being silently dropped."""
	midi = pretty_midi.PrettyMIDI(str(path))
	instrument = _select_track(midi, track)
	resolved_meter = meter or _meter_from_file(midi)
	tempo = _tempo(midi)

	eighth = Fraction(30, tempo)

	def to_eighths(seconds: float) -> Fraction:
		return snap(Fraction(seconds).limit_denominator(10_000) / eighth)

	notes = [
		Note(
			onset=to_eighths(note.start),
			duration=to_eighths(note.end) - to_eighths(note.start),
			pitch=note.pitch,
		)
		for note in instrument.notes
	]
	# A file carries no trace of how long the rests after the last note were, so
	# the melody's length is recovered as the end of the bar containing its last
	# note — the same rule a human would read off the score.
	end = max((note.onset + note.duration for note in notes), default=Fraction(0))
	bar = resolved_meter.bar_eighths
	ceiling = (end + bar - 1) // bar * bar if end else 0
	length = max(1, int(ceiling))
	# A MIDI key signature number names a key directly (0-11 major, 12-23 minor),
	# but its signature is that of the relative major — e.g. 16 is E minor with one
	# sharp. The candidate pair therefore hangs off the signature's major tonic,
	# read from the sharps count the number decodes to. External input gets a
	# single key (no modulation detection), so the first signature narrows the
	# candidates even when the file changes key later.
	signature_tonic = None
	if midi.key_signature_changes:
		_mode, sharps = pretty_midi.key_number_to_mode_accidentals(
			midi.key_signature_changes[0].key_number
		)
		signature_tonic = _MAJOR_TONIC_BY_SHARPS[sharps]
	resolved_key = (
		key if key is not None else estimate_key(tuple(notes), signature_tonic=signature_tonic)
	)
	melody = build_melody(notes, resolved_meter, (KeySpan(0, resolved_key),), length)
	return ImportedMelody(melody=melody, tempo=tempo)


_MAJOR_TONIC_BY_SHARPS = {
	-7: 11,
	-6: 6,
	-5: 1,
	-4: 8,
	-3: 3,
	-2: 10,
	-1: 5,
	0: 0,
	1: 7,
	2: 2,
	3: 9,
	4: 4,
	5: 11,
	6: 6,
	7: 1,
}
"""Major tonic pitch class for each key-signature sharps count."""


def _select_track(midi: pretty_midi.PrettyMIDI, track: int | None):
	instruments = [i for i in midi.instruments if i.notes]
	if not instruments:
		raise ValueError("no notes in any MIDI track")
	if track is not None:
		if not 0 <= track < len(midi.instruments):
			raise ValueError(f"track {track} out of range (0-{len(midi.instruments) - 1})")
		return midi.instruments[track]
	if len(instruments) == 1:
		return instruments[0]
	return max(instruments, key=_melody_score)


def _melody_score(instrument) -> float:
	"""How melody-like a track is.

	Accompaniment tracks lose points for sitting low and for playing chords: an
	onset shared by several notes is one melody note plus filler, and a highest
	line far below the file's ceiling marks a bass or chord track."""
	onsets: dict[float, list[int]] = {}
	for note in instrument.notes:
		onsets.setdefault(round(note.start, 4), []).append(note.pitch)
	pitches = [p for group in onsets.values() for p in group]
	if not pitches:
		return -1.0
	polyphonic = sum(len(group) > 1 for group in onsets.values()) / len(onsets)
	ceiling = max(pitches)
	top_line = max(max(group) for group in onsets.values())
	height = (top_line - 36) / max(1, ceiling - 36) if ceiling > 36 else 0.5
	return len(onsets) * (1.0 - polyphonic) * (0.25 + height)


def _meter_from_file(midi: pretty_midi.PrettyMIDI) -> Meter:
	signatures = midi.time_signature_changes
	if not signatures:
		raise ValueError("no time signature in MIDI; pass meter=")
	if len(signatures) > 1:
		raise ValueError("time signature changes in MIDI; split the song and pass meter=")
	signature = signatures[0]
	if signature.denominator not in (4, 8):
		raise ValueError(f"unsupported beat unit {signature.denominator}; pass meter=")
	return Meter(signature.numerator, signature.denominator)


def _tempo(midi: pretty_midi.PrettyMIDI) -> int:
	tempos = midi.get_tempo_changes()[1]
	return round(float(tempos[0])) if len(tempos) else DEFAULT_TEMPO
