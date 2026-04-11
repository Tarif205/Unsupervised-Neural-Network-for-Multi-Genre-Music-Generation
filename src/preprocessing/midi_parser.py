import pretty_midi
import numpy as np


def parse_midi(path):
    """
    Parse a MIDI file and extract note information.
    
    Args:
        path (str): Path to MIDI file
        
    Returns:
        list: List of tuples (start_time, end_time, pitch, velocity)
    """
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []
    
    notes = []
    
    # Extract notes from all instruments
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks for now
            
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    
    return notes
