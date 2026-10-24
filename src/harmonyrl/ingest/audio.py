"""Import a melody from a monophonic audio recording (WAV/MP3)."""

from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np

from harmonyrl.schema import Key, KeySpan, Meter, Note

from .common import ImportedMelody, bar_ceiling, build_melody, eighth_seconds, estimate_key, snap

FMIN, FMAX = 65.0, 2700.0
"""Pitch search range in Hz: C2 to just above E7. Dataset melodies reach MIDI 96
in 0.04% of notes (MIDI 100+ in 0.007%); a narrower cap folds those whistled
tops an octave down, a wider one costs pyin resolution."""

FRAME_MS = 20
"""pyin hop in milliseconds: the finest grid note boundaries are cut on."""

MAX_GAP_FRAMES = 1
"""Single-frame (20 ms) tracker dropouts stay inside a voiced run; longer unvoiced
stretches are rests and split runs even when the pitch before and after is identical."""

SPLIT_SEMITONES = 0.7
"""Deviation from the current note's median that starts a new note, with hysteresis."""

SPLIT_FRAMES = 3
"""Frames a deviation must persist before splitting: vibrato and glides stay one note."""

MIN_NOTE_FRAMES = 5
"""Segments shorter than this (100 ms at 20 ms frames) fold into an adjacent note
within the same voiced run, or are dropped when isolated in silence."""

PROBABILITY_MIN = 0.2
"""pyin frame-confidence floor: silence with a noise floor still shows spurious
low-frequency voiced frames, which never clear this bar."""

RMS_FLOOR_DB = -30.0
"""Frame energy floor relative to the 95th-percentile voiced RMS: pyin is amplitude-
invariant and keeps tracking periodic room/reverb decay tails well into silence."""


def melody_from_audio(
	path: Path,
	*,
	meter: Meter,
	tempo: int,
	key: Key | None = None,
) -> ImportedMelody:
	"""Read a melody from a monophonic recording.

	`meter` and `tempo` are required — a recording carries no reliable tempo
	metadata (`tempo` is dotted-quarter BPM for `x/8` meters, quarter-note BPM
	for `x/4`, matching the renderer). Frames come from pyin; contiguous voiced
	runs preserve rests, pitch steps inside a run split notes with hysteresis,
	and onsets snap to 1/48 of an 8th note."""
	eighth = eighth_seconds(tempo, meter)
	samples, sample_rate = librosa.load(str(path), sr=None, mono=True)
	hop = max(1, int(sample_rate * FRAME_MS / 1000))
	f0, voiced_flag, probability = librosa.pyin(
		samples, fmin=FMIN, fmax=FMAX, sr=sample_rate, hop_length=hop
	)
	# pyin marks low-level noise and quiet reverb decay tails as voiced because its
	# autocorrelation is amplitude-invariant; require both tracker confidence and
	# audible frame energy relative to the melody.
	rms = librosa.feature.rms(y=samples, hop_length=hop)[0][: len(f0)]
	valid_voiced = voiced_flag & ~np.isnan(f0)
	if not np.any(valid_voiced):
		raise ValueError(f"no voiced melody found in {path}")
	energy_floor = float(np.percentile(rms[valid_voiced], 95)) * (10.0 ** (RMS_FLOOR_DB / 20.0))
	tracked = valid_voiced & (probability >= PROBABILITY_MIN) & (rms >= energy_floor)
	tracked_index = np.where(tracked)[0]
	if not len(tracked_index):
		raise ValueError(f"no voiced melody found in {path}")

	def to_eighths(seconds: float) -> Fraction:
		return snap(Fraction(seconds).limit_denominator(100_000) / eighth)

	notes: list[Note] = []
	prev_end_frame = -1
	for run in _voiced_runs(tracked_index):
		for first_frame, last_frame, median in _segments_in_run(run, f0):
			# Note attacks are voiced but uncertain, so the probability filter pushes
			# every onset late; reach back over contiguous audible frames at the same
			# pitch without crossing into the previous note.
			while (
				first_frame > prev_end_frame + 1
				and valid_voiced[first_frame - 1]
				and rms[first_frame - 1] >= energy_floor
				and abs(12 * np.log2(f0[first_frame - 1] / median)) <= SPLIT_SEMITONES
			):
				first_frame -= 1
			prev_end_frame = last_frame
			onset = to_eighths(first_frame * hop / sample_rate)
			duration = to_eighths((last_frame + 1) * hop / sample_rate) - onset
			pitch = round(69 + 12 * np.log2(median / 440.0))
			if duration > 0 and 0 <= pitch <= 127:
				notes.append(Note(onset, duration, pitch))

	if not notes:
		raise ValueError(f"no voiced melody found in {path}")

	resolved_key = key if key is not None else estimate_key(tuple(notes))
	end = max(note.offset for note in notes)
	melody = build_melody(
		notes, meter, (KeySpan(0, resolved_key),), max(1, bar_ceiling(end, meter.bar_eighths))
	)
	return ImportedMelody(melody=melody, tempo=tempo)


def _voiced_runs(tracked_index: np.ndarray) -> list[np.ndarray]:
	"""Split confident voiced frame indices at unvoiced gaps longer than `MAX_GAP_FRAMES`."""
	breaks = np.where(np.diff(tracked_index) > MAX_GAP_FRAMES + 1)[0] + 1
	return list(np.split(tracked_index, breaks))


def _segments_in_run(run_frames: np.ndarray, f0: np.ndarray) -> list[tuple[int, int, float]]:
	"""Cut one contiguous voiced run into `(first_frame, last_frame, median_f0)` notes."""
	if int(run_frames[-1] - run_frames[0] + 1) < MIN_NOTE_FRAMES:
		return []

	voiced = f0[run_frames]
	raw: list[tuple[int, int]] = []
	start = 0
	for i in range(1, len(voiced) + 1):
		if i == len(voiced):
			raw.append((start, i - 1))
			break
		window = voiced[i : i + SPLIT_FRAMES]
		if len(window) < SPLIT_FRAMES:
			continue
		median = float(np.median(voiced[start:i]))
		deviations = 12 * np.log2(window / median)
		if np.all(deviations > SPLIT_SEMITONES) or np.all(deviations < -SPLIT_SEMITONES):
			raw.append((start, i - 1))
			start = i

	merged: list[tuple[int, int]] = []
	for seg_start, seg_end in raw:
		span = int(run_frames[seg_end] - run_frames[seg_start] + 1)
		if merged and span < MIN_NOTE_FRAMES:
			prev_start, _ = merged[-1]
			merged[-1] = (prev_start, seg_end)
		else:
			merged.append((seg_start, seg_end))

	if (
		len(merged) >= 2
		and int(run_frames[merged[0][1]] - run_frames[merged[0][0]] + 1) < MIN_NOTE_FRAMES
	):
		merged[1] = (merged[0][0], merged[1][1])
		merged.pop(0)

	return [
		(int(run_frames[s]), int(run_frames[e]), float(np.median(voiced[s : e + 1])))
		for s, e in merged
	]
