# src/preprocessing/midi_parser.py
import pretty_midi


def parse_midi(path):
    """
    Parse a MIDI file and extract note information.
    Includes BOTH drum and melodic instruments.

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
    for instrument in midi.instruments:
        # ── CRITICAL FIX ──────────────────────────────────────────
        # Groove MIDI is an all-drum dataset.
        # The original code skipped drum tracks, producing empty rolls.
        # We now include ALL instruments (drum + melodic).
        # ──────────────────────────────────────────────────────────
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))

    return notes


