"""Idiom statistics: non-chord tones on strong beats and how they resolve, and chord kinds.

Counts add up over segments, and `idiom_rates` turns the totals into percentages. A note is judged
against the chord at its onset, and notes under no chord are left out of the non-chord tones. A
long non-chord tone starts on a strong beat (a beat strength of 2 or more) and lasts at least 1
beat; it resolves when the next note lies within 2 semitones and is a tone of the chord at its
onset. Chord kinds count events.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

from harmonyrl.schema import ChordEvent, Melody, Note

from .spans import chord_spans, span_at


def idiom_counts(
	melody: Melody, events: Sequence[ChordEvent], *, projected: bool = True
) -> Counter[str]:
	"""Counts of notes, non-chord tones, and chord kinds in 1 segment.

	With `projected` off, chords are counted as given rather than in the model vocabulary.
	"""
	spans = chord_spans(melody, events, projected=projected)
	counts = Counter[str](segments=1, notes=len(melody.notes))

	for span in spans:
		chord = span.chord
		if chord is None:
			continue
		counts["chords"] += 1
		counts["tension_chords"] += (
			chord.type in (9, 11, 13)
			or bool(chord.suspensions)
			or bool(chord.adds)
			or bool(chord.alterations)
		)
		counts["borrowed_chords"] += chord.borrowed is not None
		counts["applied_chords"] += chord.applied != 0

	tones = [span_at(spans, note.onset).pitch_classes for note in melody.notes]
	for i, (note, pitch_classes) in enumerate(zip(melody.notes, tones, strict=True)):
		if not pitch_classes:
			continue
		counts["notes_under_chord"] += 1
		if note.pitch % 12 in pitch_classes:
			continue
		counts["non_chord_tones"] += 1
		if not _is_long_on_strong_beat(note, melody):
			continue
		counts["long_non_chord_tones"] += 1
		if i + 1 < len(melody.notes):
			following = melody.notes[i + 1]
			counts["resolved"] += (
				abs(following.pitch - note.pitch) <= 2 and following.pitch % 12 in tones[i + 1]
			)

	counts["segments_with_long"] = int(counts["long_non_chord_tones"] > 0)
	return counts


def idiom_rates(counts: Mapping[str, int]) -> dict[str, float | None]:
	"""Percentages from summed `idiom_counts`, None where the denominator is 0."""

	def percent(numerator: str, denominator: str) -> float | None:
		total = counts.get(denominator, 0)
		return 100 * counts.get(numerator, 0) / total if total else None

	return {
		"non_chord_tone_pct": percent("non_chord_tones", "notes_under_chord"),
		"long_non_chord_note_pct": percent("long_non_chord_tones", "notes"),
		"long_non_chord_segment_pct": percent("segments_with_long", "segments"),
		"resolution_pct": percent("resolved", "long_non_chord_tones"),
		"tension_pct": percent("tension_chords", "chords"),
		"borrowed_pct": percent("borrowed_chords", "chords"),
		"applied_pct": percent("applied_chords", "chords"),
	}


def _is_long_on_strong_beat(note: Note, melody: Melody) -> bool:
	if note.onset.denominator != 1:
		return False
	strong = melody.meter.beat_strength(int(note.onset)) >= 2
	return strong and note.duration >= melody.meter.eighths_per_beat
