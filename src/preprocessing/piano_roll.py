# src/preprocessing/piano_roll.py
import numpy as np


def notes_to_piano_roll(notes, max_len=500, fs=100):
    """
    Convert note events to piano roll representation.

    Args:
        notes (list): List of tuples (start_time, end_time, pitch, velocity)
        max_len (int): Maximum length in frames
        fs (int): Frames per second

    Returns:
        np.ndarray: Shape (max_len, 88)  i.e. (T, pitch)
                    Raw velocities 0-127 — normalisation happens in dataset.py
    """
    # ── CRITICAL FIX ──────────────────────────────────────────────
    # Original returned (88, max_len) — pitch-major order.
    # All downstream code (dataset, model) expects (T, 88) — time-major.
    # Axes were swapped, so the model was reading pitch as time.
    # ──────────────────────────────────────────────────────────────
    piano_roll = np.zeros((max_len, 88), dtype=np.float32)

    if not notes:
        return piano_roll

    for start, end, pitch, velocity in notes:
        start_frame = int(start * fs)
        end_frame   = int(end   * fs)

        # MIDI pitch 21–108  →  index 0–87
        pitch_idx = max(0, min(87, pitch - 21))

        end_frame = min(end_frame, max_len)

        if start_frame < max_len:
            piano_roll[start_frame:end_frame, pitch_idx] = velocity

    return piano_roll   # (T, 88)
