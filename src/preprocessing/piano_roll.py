import numpy as np


def notes_to_piano_roll(notes, max_len=500, fs=100):
    """
    Convert note events to piano roll representation.
    
    Args:
        notes (list): List of tuples (start_time, end_time, pitch, velocity)
        max_len (int): Maximum length of piano roll in frames
        fs (int): Frame sampling rate (frames per second)
        
    Returns:
        np.ndarray: Piano roll of shape (88, max_len) - 88 piano keys
    """
    piano_roll = np.zeros((88, max_len))
    
    if not notes:
        return piano_roll
    
    for start, end, pitch, velocity in notes:
        start_frame = int(start * fs)
        end_frame = int(end * fs)
        
        # MIDI pitch 21-108 maps to piano keys 0-87
        pitch_idx = max(0, min(87, pitch - 21))
        
        # Ensure we don't exceed max_len
        end_frame = min(end_frame, max_len)
        
        if start_frame < max_len:
            piano_roll[pitch_idx, start_frame:end_frame] = velocity
    
    return piano_roll
