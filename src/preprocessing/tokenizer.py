import numpy as np


def tokenize(piano_roll):
    """
    Convert piano roll to tokens for model input.
    
    Args:
        piano_roll (np.ndarray): Piano roll array of shape (88, seq_len)
        
    Returns:
        np.ndarray: Tokenized representation (flattened or encoded)
    """
    # Flatten the piano roll into a sequence of velocity values
    tokens = piano_roll.flatten().astype(np.float32)
    return tokens
