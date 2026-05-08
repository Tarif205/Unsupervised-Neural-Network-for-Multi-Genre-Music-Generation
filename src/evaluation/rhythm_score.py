# src/evaluation/rhythm_score.py
"""
Rhythm Diversity Score
D_rhythm = #unique durations / #total notes

Faculty guide:
- Compute duration as end - start in seconds
- Quantize to nearest 50ms to avoid floating point noise
- Divide unique quantized durations by total note count
- Higher = more rhythmically diverse
"""

import numpy as np
import pretty_midi
from collections import Counter


def quantize_duration(duration_seconds: float, resolution_ms: float = 50.0) -> int:
    """Quantize duration to nearest 50ms bucket."""
    resolution_s = resolution_ms / 1000.0
    return int(round(duration_seconds / resolution_s))


def rhythm_diversity_from_midi(midi_path: str) -> float:
    """
    D_rhythm = #unique durations / #total notes
    Computed from a MIDI file.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return 0.0

    durations = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            dur = note.end - note.start
            if dur > 0:
                durations.append(quantize_duration(dur))

    if not durations:
        return 0.0

    unique = len(set(durations))
    total  = len(durations)
    return unique / total


def rhythm_diversity_from_roll(piano_roll: np.ndarray,
                                fs: int = 16,
                                threshold: float = 0.15) -> float:
    """
    D_rhythm computed directly from piano roll (T, 88).
    """
    durations = []
    T = piano_roll.shape[0]

    for pitch_idx in range(piano_roll.shape[1]):
        in_note    = False
        note_start = 0

        for t in range(T):
            active = piano_roll[t, pitch_idx] > threshold

            if active and not in_note:
                in_note    = True
                note_start = t
            elif not active and in_note:
                dur_seconds = (t - note_start) / fs
                durations.append(quantize_duration(dur_seconds))
                in_note = False

        if in_note:
            dur_seconds = (T - note_start) / fs
            durations.append(quantize_duration(dur_seconds))

    if not durations:
        return 0.0

    return len(set(durations)) / len(durations)


def repetition_ratio_from_midi(midi_path: str, ngram_size: int = 4) -> float:
    """
    R = #repeated patterns / #total patterns

    Faculty guide:
    - Extract sorted-by-onset pitch sequence
    - Compute overlapping n-grams of length 4
    - Count n-grams appearing more than once
    - Values 0.1-0.5 typical for coherent classical music
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return 0.0

    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.pitch))

    # Sort by onset time
    notes.sort(key=lambda x: x[0])
    pitch_seq = [n[1] for n in notes]

    if len(pitch_seq) < ngram_size:
        return 0.0

    # Compute overlapping n-grams
    ngrams = [
        tuple(pitch_seq[i:i + ngram_size])
        for i in range(len(pitch_seq) - ngram_size + 1)
    ]

    if not ngrams:
        return 0.0

    counts         = Counter(ngrams)
    repeated_count = sum(v for v in counts.values() if v > 1)
    return repeated_count / len(ngrams)