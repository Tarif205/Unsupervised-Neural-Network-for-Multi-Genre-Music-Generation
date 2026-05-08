# src/preprocessing/tokenizer.py
import numpy as np


def tokenize(piano_roll):
    """
    Flatten a piano roll (T, 88) into a 1-D array for storage.

    Note: this is a storage helper, NOT discrete tokenisation.
    The name is kept for backward compatibility.
    Task 3 (Transformer) uses a separate event-token encoder.

    Args:
        piano_roll (np.ndarray): shape (T, 88), raw velocities 0-127

    Returns:
        np.ndarray: shape (T*88,)  dtype float32
    """
    # Flatten row-major: frame0_pitch0, frame0_pitch1, ..., frameT_pitch87
    # This preserves time-major order so reshape back to (T, 88) is correct.
    return piano_roll.flatten().astype(np.float32)


