

# src/generation/midi_export.py

import pretty_midi
import numpy as np


def piano_roll_to_midi(piano_roll, output_path, fs=16, threshold=0.05):

    piano_roll = np.array(piano_roll)

    midi = pretty_midi.PrettyMIDI()

    # IMPORTANT: drum mode
    instrument = pretty_midi.Instrument(program=0, is_drum=True)

    T, P = piano_roll.shape

    MAX_NOTE_LEN = 10   # controls note splitting

    for pitch in range(P):
        active = False
        start = 0

        for t in range(T):
            if piano_roll[t, pitch] > threshold:
                if not active:
                    active = True
                    start = t
            else:
                if active:
                    end = t
                    active = False

                    cur = start
                    while cur < end:
                        split_end = min(cur + MAX_NOTE_LEN, end)

                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=pitch + 21,
                            start=cur / fs,
                            end=split_end / fs
                        )
                        instrument.notes.append(note)

                        cur = split_end

        if active:
            end = T
            cur = start
            while cur < end:
                split_end = min(cur + MAX_NOTE_LEN, end)

                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch + 21,
                    start=cur / fs,
                    end=split_end / fs
                )
                instrument.notes.append(note)

                cur = split_end

    midi.instruments.append(instrument)
    midi.write(output_path)