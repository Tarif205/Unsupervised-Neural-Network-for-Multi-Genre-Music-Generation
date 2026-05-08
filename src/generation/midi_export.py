# #src/generation/midi_export.py
# import pretty_midi
# import numpy as np


# def piano_roll_to_midi(piano_roll, output_path, fs=100, threshold=0.15, program=0, is_drum=False):
#     """
#     Convert a piano roll of shape (T, 88) to a MIDI file.

#     piano_roll values are expected to be in [0, 1].
#     """
#     piano_roll = np.asarray(piano_roll, dtype=np.float32)
#     piano_roll = np.clip(piano_roll, 0.0, 1.0)

#     active = piano_roll >= threshold
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)

#     time_per_frame = 1.0 / float(fs)
#     total_frames = piano_roll.shape[0]

#     for pitch_index in range(piano_roll.shape[1]):
#         in_note = False
#         start_frame = 0

#         for frame in range(total_frames):
#             is_active = bool(active[frame, pitch_index])

#             if is_active and not in_note:
#                 in_note = True
#                 start_frame = frame
#             elif not is_active and in_note:
#                 end_frame = frame
#                 velocity_value = float(np.max(piano_roll[start_frame:end_frame, pitch_index]))
#                 velocity = int(max(1, min(127, round(velocity_value * 127))))
#                 note = pretty_midi.Note(
#                     velocity=velocity,
#                     pitch=pitch_index + 21,
#                     start=start_frame * time_per_frame,
#                     end=end_frame * time_per_frame,
#                 )
#                 instrument.notes.append(note)
#                 in_note = False

#         if in_note:
#             end_frame = total_frames
#             velocity_value = float(np.max(piano_roll[start_frame:end_frame, pitch_index]))
#             velocity = int(max(1, min(127, round(velocity_value * 127))))
#             note = pretty_midi.Note(
#                 velocity=velocity,
#                 pitch=pitch_index + 21,
#                 start=start_frame * time_per_frame,
#                 end=end_frame * time_per_frame,
#             )
#             instrument.notes.append(note)

#     midi.instruments.append(instrument)
#     midi.write(output_path)
# src/generation/midi_export.py

# import pretty_midi
# import numpy as np


# def piano_roll_to_midi(piano_roll, output_path, fs=16, threshold=0.05):
#     """
#     Convert piano roll (T x 88) to MIDI file.

#     Fixes:
#     - Handles thresholding properly
#     - Splits long notes into smaller segments (prevents low note count)
#     """

#     piano_roll = np.array(piano_roll)

#     # Create PrettyMIDI object
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

#     T, P = piano_roll.shape
#     MAX_NOTE_LEN = 10  # frames (controls note splitting)

#     for pitch in range(P):
#         active = False
#         start = 0

#         for t in range(T):
#             val = piano_roll[t, pitch]

#             if val > threshold:
#                 if not active:
#                     active = True
#                     start = t
#             else:
#                 if active:
#                     end = t
#                     active = False

#                     # Split long notes
#                     cur = start
#                     while cur < end:
#                         split_end = min(cur + MAX_NOTE_LEN, end)

#                         note = pretty_midi.Note(
#                             velocity=80,
#                             pitch=pitch + 21,  # MIDI mapping (A0=21)
#                             start=cur / fs,
#                             end=split_end / fs
#                         )
#                         instrument.notes.append(note)

#                         cur = split_end

#         # Handle case if note continues till end
#         if active:
#             end = T
#             cur = start
#             while cur < end:
#                 split_end = min(cur + MAX_NOTE_LEN, end)

#                 note = pretty_midi.Note(
#                     velocity=80,
#                     pitch=pitch + 21,
#                     start=cur / fs,
#                     end=split_end / fs
#                 )
#                 instrument.notes.append(note)

#                 cur = split_end

#     midi.instruments.append(instrument)
#     midi.write(output_path)


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