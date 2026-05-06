# # src/preprocessing/tokenize_midi.py
# """
# Task 3 Preprocessing — MIDI → REMI Token Sequences
# Faculty guide: "Select the REMI tokenizer in miditok"
#                "Each file → integer ID list"
#                "Save token sequences for Transformer training"
# """

# import os
# import sys
# import json
# import pickle
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from miditok import REMI, TokenizerConfig
# from miditok.utils import get_midi_programs
# from config import (
#     RAW_DIR, PROCESSED_DIR, SEED,
#     TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
#     TF_MAX_LEN
# )

# # ── Tokenizer config ───────────────────────────────────────────
# TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "remi_tokenizer.json")
# TOKEN_DIR      = os.path.join(PROCESSED_DIR, "tokens")
# MAX_SEQ_LEN    = TF_MAX_LEN   # 512 tokens per sequence


# def build_tokenizer():
#     """Build and save REMI tokenizer."""
#     config = TokenizerConfig(
#         num_velocities    = 32,    # 32 velocity bins
#         use_chords        = False,
#         use_rests         = True,
#         use_tempos        = True,
#         use_time_signatures = False,
#         use_programs      = False,
#     )
#     tokenizer = REMI(config)
#     print(f"Vocabulary size: {len(tokenizer.vocab)}")
#     return tokenizer


# def collect_midi_files(raw_dir):
#     midi_files = []
#     for root, _, files in os.walk(raw_dir):
#         for f in files:
#             if f.lower().endswith(('.mid', '.midi')):
#                 midi_files.append(os.path.join(root, f))
#     midi_files.sort()
#     return midi_files


# def split_files(midi_files):
#     """File-level split to prevent leakage."""
#     import random
#     random.seed(SEED)
#     files = midi_files.copy()
#     random.shuffle(files)

#     n       = len(files)
#     n_train = int(n * TRAIN_RATIO)
#     n_val   = int(n * VAL_RATIO)

#     train = files[:n_train]
#     val   = files[n_train:n_train + n_val]
#     test  = files[n_train + n_val:]
#     return train, val, test


# def tokenize_file(tokenizer, midi_path, max_seq_len):
#     """
#     Tokenize one MIDI file → list of fixed-length sequences.
#     Returns list of np.array each shape (max_seq_len,)
#     """
#     try:
#         tokens = tokenizer(midi_path)
#         if not tokens or not tokens[0].ids:
#             return []

#         ids = tokens[0].ids
#         sequences = []

#         # Slide window over token sequence
#         for start in range(0, len(ids) - max_seq_len, max_seq_len // 2):
#             seq = ids[start: start + max_seq_len]
#             if len(seq) == max_seq_len:
#                 sequences.append(np.array(seq, dtype=np.int32))

#         return sequences
#     except Exception as e:
#         return []


# def build_split(tokenizer, file_list, split_name, max_seq_len):
#     """Tokenize all files in a split and save."""
#     all_seqs = []
#     skipped  = 0

#     for path in tqdm(file_list, desc=f"Tokenizing {split_name}"):
#         seqs = tokenize_file(tokenizer, path, max_seq_len)
#         if seqs:
#             all_seqs.extend(seqs)
#         else:
#             skipped += 1

#     if not all_seqs:
#         print(f"  WARNING: No sequences for {split_name}")
#         return np.empty((0, max_seq_len), dtype=np.int32)

#     arr = np.stack(all_seqs, axis=0)
#     print(f"  {split_name}: {arr.shape} | skipped: {skipped}")
#     return arr


# def main():
#     os.makedirs(PROCESSED_DIR, exist_ok=True)
#     os.makedirs(TOKEN_DIR, exist_ok=True)

#     print("Building REMI tokenizer...")
#     tokenizer = build_tokenizer()

#     # Save tokenizer
#     tokenizer.save(Path(TOKENIZER_PATH))
#     print(f"Tokenizer saved → {TOKENIZER_PATH}")

#     # Collect MIDI files
#     midi_files = collect_midi_files(RAW_DIR)
#     print(f"\nMIDI files found: {len(midi_files)}")

#     train_files, val_files, test_files = split_files(midi_files)
#     print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

#     # Tokenize each split
#     print(f"\nTokenizing with MAX_SEQ_LEN={MAX_SEQ_LEN}...")

#     train_arr = build_split(tokenizer, train_files, "train", MAX_SEQ_LEN)
#     val_arr   = build_split(tokenizer, val_files,   "val",   MAX_SEQ_LEN)
#     test_arr  = build_split(tokenizer, test_files,  "test",  MAX_SEQ_LEN)

#     # Save
#     np.save(os.path.join(TOKEN_DIR, "train_tokens.npy"), train_arr)
#     np.save(os.path.join(TOKEN_DIR, "val_tokens.npy"),   val_arr)
#     np.save(os.path.join(TOKEN_DIR, "test_tokens.npy"),  test_arr)

#     # Save metadata
#     meta = {
#         "vocab_size"  : len(tokenizer.vocab),
#         "max_seq_len" : MAX_SEQ_LEN,
#         "train_shape" : list(train_arr.shape),
#         "val_shape"   : list(val_arr.shape),
#         "test_shape"  : list(test_arr.shape),
#         "tokenizer"   : "REMI",
#         "num_velocities": 32,
#     }
#     with open(os.path.join(TOKEN_DIR, "metadata.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     print(f"\n✅ Tokenization complete!")
#     print(f"   Train : {train_arr.shape}")
#     print(f"   Val   : {val_arr.shape}")
#     print(f"   Test  : {test_arr.shape}")
#     print(f"   Vocab : {len(tokenizer.vocab)}")
#     print(f"   Saved → {TOKEN_DIR}")


# if __name__ == "__main__":
#     main()

# src/preprocessing/tokenize_midi.py

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from miditok import REMI, TokenizerConfig
from config import (
    RAW_DIR, PROCESSED_DIR, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TF_MAX_LEN
)

# Paths
TOKENIZER_PATH = os.path.join(PROCESSED_DIR, "remi_tokenizer.json")
TOKEN_DIR = os.path.join(PROCESSED_DIR, "tokens")
MAX_SEQ_LEN = TF_MAX_LEN


def build_tokenizer():
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=False,
    )
    tokenizer = REMI(config)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    return tokenizer


def collect_midi_files(raw_dir):
    midi_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))
    midi_files.sort()
    return midi_files


def split_files(midi_files):
    import random
    random.seed(SEED)

    files = midi_files.copy()
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    return train, val, test


def tokenize_file(tokenizer, midi_path, max_seq_len):
    try:
        tokens = tokenizer(midi_path)

        if not tokens or len(tokens) == 0 or len(tokens[0].ids) == 0:
            return []

        ids = tokens[0].ids
        sequences = []

        stride = max_seq_len // 2

        for start in range(0, len(ids) - max_seq_len, stride):
            seq = ids[start:start + max_seq_len]

            if len(seq) == max_seq_len:
                sequences.append(np.array(seq, dtype=np.int32))

        return sequences

    except Exception as e:
        print(f"Skipped {midi_path} | Error: {e}")
        return []


def build_split(tokenizer, file_list, split_name, max_seq_len):
    all_seqs = []
    skipped = 0

    for path in tqdm(file_list, desc=f"Tokenizing {split_name}"):
        seqs = tokenize_file(tokenizer, path, max_seq_len)

        if seqs:
            all_seqs.extend(seqs)
        else:
            skipped += 1

    if not all_seqs:
        print(f"WARNING: No sequences for {split_name}")
        return np.empty((0, max_seq_len), dtype=np.int32)

    arr = np.stack(all_seqs, axis=0)
    print(f"{split_name}: {arr.shape} | skipped: {skipped}")

    return arr


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TOKEN_DIR, exist_ok=True)

    print("Building REMI tokenizer...")
    tokenizer = build_tokenizer()

    tokenizer.save(Path(TOKENIZER_PATH))
    print(f"Tokenizer saved → {TOKENIZER_PATH}")

    midi_files = collect_midi_files(RAW_DIR)
    print(f"\nMIDI files found: {len(midi_files)}")

    train_files, val_files, test_files = split_files(midi_files)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print(f"\nTokenizing with MAX_SEQ_LEN={MAX_SEQ_LEN}...")

    train_arr = build_split(tokenizer, train_files, "train", MAX_SEQ_LEN)
    val_arr = build_split(tokenizer, val_files, "val", MAX_SEQ_LEN)
    test_arr = build_split(tokenizer, test_files, "test", MAX_SEQ_LEN)

    np.save(os.path.join(TOKEN_DIR, "train_tokens.npy"), train_arr)
    np.save(os.path.join(TOKEN_DIR, "val_tokens.npy"), val_arr)
    np.save(os.path.join(TOKEN_DIR, "test_tokens.npy"), test_arr)

    meta = {
        "vocab_size": len(tokenizer.vocab),
        "max_seq_len": MAX_SEQ_LEN,
        "train_shape": list(train_arr.shape),
        "val_shape": list(val_arr.shape),
        "test_shape": list(test_arr.shape),
        "tokenizer": "REMI",
        "num_velocities": 32,
    }

    with open(os.path.join(TOKEN_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nTokenization complete!")
    print(f"Train : {train_arr.shape}")
    print(f"Val   : {val_arr.shape}")
    print(f"Test  : {test_arr.shape}")
    print(f"Vocab : {len(tokenizer.vocab)}")
    print(f"Saved → {TOKEN_DIR}")


if __name__ == "__main__":
    main()