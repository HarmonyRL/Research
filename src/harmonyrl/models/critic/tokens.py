"""Critic inputs: a harmonization played with the fixed voicing, as REMI+BPE token ids."""

from collections.abc import Sequence

import numpy as np
from miditok import MusicTokenizer
from symusic.core import NoteTick, ScoreTick, TempoTick, TimeSignatureTick, TrackTick

from harmonyrl.render import DEFAULT_TEMPO, chord_notes, melody_notes
from harmonyrl.schema import ChordEvent, Melody

TICKS_PER_QUARTER = 480
"""Score resolution. 240 ticks per 8th note keep every note on the 1/48 grid exact."""


def harmonization_score(melody: Melody, events: Sequence[ChordEvent]) -> ScoreTick:
	"""The MIDI that `harmonyrl.render.write_midi` writes for `melody` with `events`, in memory."""
	tempo = DEFAULT_TEMPO if melody.tempo is None else melody.tempo
	meter = melody.meter
	score = ScoreTick(TICKS_PER_QUARTER)
	score.tempos.append(TempoTick(0, qpm=tempo * (3 if meter.beat_unit == 8 else 2) / 2))
	score.time_signatures.append(TimeSignatureTick(0, meter.beats_per_bar, meter.beat_unit))
	eighth = TICKS_PER_QUARTER // 2

	for notes in (melody_notes(melody), chord_notes(melody, events)):
		if not notes:
			continue

		track = TrackTick("", notes[0].program)

		# Notes read from a MIDI file come ordered by time and pitch, and the tokenizer keepts the
		# order it is given, so the voicing order would otherwise change the tokens.
		for note in sorted(notes, key=lambda n: (n.onset, n.pitch)):
			track.notes.append(
				NoteTick(
					round(note.onset * eighth),
					round(note.duration * eighth),
					note.pitch,
					note.velocity,
				)
			)

		score.tracks.append(track)

	return score


def token_ids(
	tokenizer: MusicTokenizer,
	melody: Melody,
	events: Sequence[ChordEvent],
	max_tokens: int,
) -> np.ndarray:
	"""The first `max_tokens` token ids of `melody` with `events`."""
	tokens = tokenizer.encode(harmonization_score(melody, events))
	if isinstance(tokens, list):
		raise TypeError("the tokenizer must put every track in 1 token stream")

	return np.array(tokens.ids[:max_tokens], dtype=np.int64)
