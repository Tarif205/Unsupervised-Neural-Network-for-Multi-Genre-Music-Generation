# src/evaluation/pitch_histogram.py
"""
Pitch Histogram Similarity
H(p, q) = Σ_{i=1}^{12} |p_i - q_i|

Faculty guide:
- Map every note to pitch class via pitch % 12
- Accumulate counts in 12-element array
- Normalize by total note count
- Compute L1 distance between generated and reference
- Score 0 = identical, max 2 = completely different
"""

import numpy as np
import pretty_midi


def get_pitch_class_histogram(midi_path: str) -> np.ndarray:
    """
    Compute normalized 12-dim pitch class histogram from a MIDI file.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return np.zeros(12)

    hist = np.zeros(12)
    total_notes = 0

    for instrument in midi.instruments:
        for note in instrument.notes:
            pitch_class = note.pitch % 12
            hist[pitch_class] += 1
            total_notes += 1

    if total_notes == 0:
        return hist

    return hist / total_notes


def pitch_histogram_similarity(generated_path: str, reference_path: str) -> float:
    """
    H(p, q) = Σ|p_i - q_i|
    Lower = more similar to real music.
    """
    p = get_pitch_class_histogram(reference_path)
    q = get_pitch_class_histogram(generated_path)
    return float(np.sum(np.abs(p - q)))


def pitch_histogram_from_roll(piano_roll: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Compute pitch class histogram directly from piano roll (T, 88).
    """
    hist  = np.zeros(12)
    active = (piano_roll > threshold)

    for pitch_idx in range(88):
        midi_pitch  = pitch_idx + 21
        pitch_class = midi_pitch % 12
        hist[pitch_class] += active[:, pitch_idx].sum()

    total = hist.sum()
    return hist / total if total > 0 else hist


def pitch_similarity_from_rolls(roll1: np.ndarray, roll2: np.ndarray) -> float:
    """Compare two piano rolls directly."""
    p = pitch_histogram_from_roll(roll1)
    q = pitch_histogram_from_roll(roll2)
    return float(np.sum(np.abs(p - q)))