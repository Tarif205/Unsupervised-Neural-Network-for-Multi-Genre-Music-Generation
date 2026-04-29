#src/generation/midi_export.py
import pretty_midi
import numpy as np


def piano_roll_to_midi(piano_roll, output_path, fs=100, threshold=0.15, program=0, is_drum=False):
    """
    Convert a piano roll of shape (T, 88) to a MIDI file.

    piano_roll values are expected to be in [0, 1].
    """
    piano_roll = np.asarray(piano_roll, dtype=np.float32)
    piano_roll = np.clip(piano_roll, 0.0, 1.0)

    active = piano_roll >= threshold
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)

    time_per_frame = 1.0 / float(fs)
    total_frames = piano_roll.shape[0]

    for pitch_index in range(piano_roll.shape[1]):
        in_note = False
        start_frame = 0

        for frame in range(total_frames):
            is_active = bool(active[frame, pitch_index])

            if is_active and not in_note:
                in_note = True
                start_frame = frame
            elif not is_active and in_note:
                end_frame = frame
                velocity_value = float(np.max(piano_roll[start_frame:end_frame, pitch_index]))
                velocity = int(max(1, min(127, round(velocity_value * 127))))
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch_index + 21,
                    start=start_frame * time_per_frame,
                    end=end_frame * time_per_frame,
                )
                instrument.notes.append(note)
                in_note = False

        if in_note:
            end_frame = total_frames
            velocity_value = float(np.max(piano_roll[start_frame:end_frame, pitch_index]))
            velocity = int(max(1, min(127, round(velocity_value * 127))))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch_index + 21,
                start=start_frame * time_per_frame,
                end=end_frame * time_per_frame,
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_path)
