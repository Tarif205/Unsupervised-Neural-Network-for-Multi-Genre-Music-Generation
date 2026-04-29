# # src/preprocessing/main.py
# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# from midi_parser import parse_midi
# from piano_roll import notes_to_piano_roll
# from tokenizer import tokenize


# # ======================
# # CONFIG (SAFE PATHS)
# # ======================
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))

# DATA_PATH = os.path.join(BASE_DIR, "data", "raw_midi")
# OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed")

# MAX_LEN = 500
# FS = 100


# print("🚀 Script loaded successfully")


# # ======================
# # PROCESS SINGLE FILE
# # ======================
# def process_file(path):
#     notes = parse_midi(path)
#     piano_roll = notes_to_piano_roll(notes, max_len=MAX_LEN, fs=FS)
#     tokens = tokenize(piano_roll)
#     return tokens


# # ======================
# # LOAD DATASET
# # ======================
# def load_dataset():
#     data = []
#     midi_files = []

#     print("🔍 Scanning dataset at:", DATA_PATH)

#     if not os.path.exists(DATA_PATH):
#         print("❌ DATA_PATH does not exist!")
#         return np.array([])

#     for root, _, files in os.walk(DATA_PATH):
#         for file in files:
#             if file.endswith(".mid") or file.endswith(".midi"):
#                 midi_files.append(os.path.join(root, file))

#     print(f"🎵 Total MIDI files found: {len(midi_files)}")

#     if len(midi_files) == 0:
#         print("❌ No MIDI files found. Check dataset location.")
#         return np.array([])

#     for path in tqdm(midi_files, desc="Processing MIDI"):
#         try:
#             sample = process_file(path)
#             data.append(sample)
#         except Exception as e:
#             print("❌ Error processing:", path)

#     return np.array(data)


# # ======================
# # MAIN PIPELINE
# # ======================
# def main():
#     print("🚀 Starting preprocessing pipeline...")

#     os.makedirs(OUTPUT_PATH, exist_ok=True)

#     data = load_dataset()

#     if len(data) == 0:
#         print("❌ No data processed. Exiting.")
#         return

#     print("\n📊 Dataset shape:", data.shape)

#     # Split dataset
#     train, temp = train_test_split(data, test_size=0.2, random_state=42)
#     val, test = train_test_split(temp, test_size=0.5, random_state=42)

#     print("\n📦 Data Split:")
#     print("Train:", train.shape)
#     print("Val:  ", val.shape)
#     print("Test: ", test.shape)

#     # Save
#     np.save(os.path.join(OUTPUT_PATH, "train.npy"), train)
#     np.save(os.path.join(OUTPUT_PATH, "val.npy"), val)
#     np.save(os.path.join(OUTPUT_PATH, "test.npy"), test)

#     print("\n✅ Preprocessing COMPLETE!")
#     print("Saved at:", OUTPUT_PATH)


# # ======================
# # ENTRY POINT
# # ======================
# if __name__ == "__main__":
#     main()



# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# import json
# import numpy as np
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# from midi_parser import parse_midi
# from piano_roll import notes_to_piano_roll
# from tokenizer import tokenize

# from config import (
#     RAW_DIR, PROCESSED_DIR,
#     SEQ_LEN, FEATURE_SIZE, MIDI_FS, STRIDE,
#     MIN_ACTIVE_NOTES, MIN_SEQ_COVERAGE, SKIP_EMPTY_WINDOWS,
#     TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
#     DATASET_NAME, DATASET_SUBSET, REPRESENTATION
# )

# print(f"Dataset   : {DATASET_NAME}")
# print(f"Subset    : {DATASET_SUBSET}")
# print(f"Repr      : {REPRESENTATION}")
# print(f"SEQ_LEN   : {SEQ_LEN} | STRIDE: {STRIDE} | FS: {MIDI_FS}")


# def passes_quality_filter(window: np.ndarray) -> bool:
#     """
#     window: (T, 88), raw velocities 0-127
#     """
#     if not SKIP_EMPTY_WINDOWS:
#         return True

#     active_frames = (window > 0).any(axis=1).sum()
#     coverage = active_frames / SEQ_LEN

#     if coverage < MIN_SEQ_COVERAGE:
#         return False

#     note_ons = (window > 0).sum()
#     if note_ons < MIN_ACTIVE_NOTES:
#         return False

#     return True


# def process_file(path: str):
#     """
#     Parse one MIDI file and extract multiple fixed-length windows
#     using sliding window with STRIDE.

#     Returns:
#         list[np.ndarray]: each item shape (SEQ_LEN * FEATURE_SIZE,)
#     """
#     notes = parse_midi(path)
#     if not notes:
#         return []

#     max_time = max(end for _, end, _, _ in notes)
#     total_frames = int(max_time * MIDI_FS) + 1

#     if total_frames < SEQ_LEN:
#         return []

#     full_roll = notes_to_piano_roll(notes, max_len=total_frames, fs=MIDI_FS)
#     # full_roll shape: (total_frames, 88)

#     windows = []
#     start = 0

#     while start + SEQ_LEN <= total_frames:
#         window = full_roll[start:start + SEQ_LEN]   # (SEQ_LEN, 88)

#         if passes_quality_filter(window):
#             windows.append(tokenize(window))  # flattened for storage

#         start += STRIDE

#     return windows


# def collect_midi_files(raw_dir: str):
#     midi_files = []
#     for root, _, files in os.walk(raw_dir):
#         for f in files:
#             if f.lower().endswith((".mid", ".midi")):
#                 midi_files.append(os.path.join(root, f))
#     midi_files.sort()
#     return midi_files


# def split_files(midi_files):
#     """
#     Split at FILE LEVEL to avoid leakage.
#     """
#     train_files, temp_files = train_test_split(
#         midi_files,
#         test_size=(1.0 - TRAIN_RATIO),
#         random_state=SEED
#     )

#     val_fraction_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

#     val_files, test_files = train_test_split(
#         temp_files,
#         test_size=(1.0 - val_fraction_in_temp),
#         random_state=SEED
#     )

#     return train_files, val_files, test_files


# def build_split_dataset(file_list, split_name="train"):
#     """
#     Process a list of MIDI files into a dataset array.
#     """
#     data = []
#     skipped_files = 0

#     for path in tqdm(file_list, desc=f"Processing {split_name}"):
#         try:
#             windows = process_file(path)
#             data.extend(windows)
#         except Exception as e:
#             print(f"Error processing {path}: {e}")
#             skipped_files += 1

#     if len(data) == 0:
#         print(f"WARNING: No usable windows found for {split_name}.")
#         return np.empty((0, SEQ_LEN * FEATURE_SIZE), dtype=np.float32), skipped_files

#     return np.array(data, dtype=np.float32), skipped_files


# def save_metadata(train_files, val_files, test_files, train_data, val_data, test_data):
#     metadata = {
#         "dataset_name": DATASET_NAME,
#         "dataset_subset": DATASET_SUBSET,
#         "representation": REPRESENTATION,
#         "seq_len": SEQ_LEN,
#         "feature_size": FEATURE_SIZE,
#         "midi_fs": MIDI_FS,
#         "stride": STRIDE,
#         "min_active_notes": MIN_ACTIVE_NOTES,
#         "min_seq_coverage": MIN_SEQ_COVERAGE,
#         "skip_empty_windows": SKIP_EMPTY_WINDOWS,
#         "train_ratio": TRAIN_RATIO,
#         "val_ratio": VAL_RATIO,
#         "test_ratio": TEST_RATIO,
#         "seed": SEED,
#         "num_train_files": len(train_files),
#         "num_val_files": len(val_files),
#         "num_test_files": len(test_files),
#         "train_shape": list(train_data.shape),
#         "val_shape": list(val_data.shape),
#         "test_shape": list(test_data.shape),
#     }

#     meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
#     with open(meta_path, "w") as f:
#         json.dump(metadata, f, indent=2)


# def main():
#     print("Starting preprocessing pipeline...")
#     os.makedirs(PROCESSED_DIR, exist_ok=True)

#     if not os.path.exists(RAW_DIR):
#         print("ERROR: RAW_DIR does not exist:", RAW_DIR)
#         return

#     midi_files = collect_midi_files(RAW_DIR)

#     print(f"\nScanning: {RAW_DIR}")
#     print(f"MIDI files found: {len(midi_files)}")

#     if not midi_files:
#         print("ERROR: No MIDI files found.")
#         return

#     # FILE-LEVEL split
#     train_files, val_files, test_files = split_files(midi_files)

#     print("\nFile split:")
#     print(f"  Train files: {len(train_files)}")
#     print(f"  Val files  : {len(val_files)}")
#     print(f"  Test files : {len(test_files)}")

#     # Build datasets split-wise
#     train_data, train_skipped = build_split_dataset(train_files, "train")
#     val_data, val_skipped = build_split_dataset(val_files, "val")
#     test_data, test_skipped = build_split_dataset(test_files, "test")

#     print("\nWindow datasets:")
#     print(f"  Train: {train_data.shape}")
#     print(f"  Val  : {val_data.shape}")
#     print(f"  Test : {test_data.shape}")

#     total_skipped = train_skipped + val_skipped + test_skipped
#     print(f"\nFiles with processing errors: {total_skipped}")

#     np.save(os.path.join(PROCESSED_DIR, "train.npy"), train_data)
#     np.save(os.path.join(PROCESSED_DIR, "val.npy"), val_data)
#     np.save(os.path.join(PROCESSED_DIR, "test.npy"), test_data)

#     save_metadata(train_files, val_files, test_files, train_data, val_data, test_data)

#     print(f"\nPreprocessing COMPLETE — saved to {PROCESSED_DIR}")
#     print("Saved files:")
#     print("  - train.npy")
#     print("  - val.npy")
#     print("  - test.npy")
#     print("  - metadata.json")


# if __name__ == "__main__":
#     main()

# src/preprocessing/main.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from midi_parser import parse_midi
from piano_roll import notes_to_piano_roll

from config import (
    RAW_DIR, PROCESSED_DIR,
    SEQ_LEN, FEATURE_SIZE, MIDI_FS, STRIDE,
    MIN_ACTIVE_NOTES, MIN_SEQ_COVERAGE, SKIP_EMPTY_WINDOWS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
    DATASET_NAME, DATASET_SUBSET, REPRESENTATION
)

print(f"Dataset   : {DATASET_NAME}")
print(f"Subset    : {DATASET_SUBSET}")
print(f"Repr      : {REPRESENTATION}")
print(f"SEQ_LEN   : {SEQ_LEN} | STRIDE: {STRIDE} | FS: {MIDI_FS}")


# ─────────────────────────────────────────────────────────────
# Quality filter
# ─────────────────────────────────────────────────────────────
def passes_quality_filter(window: np.ndarray) -> bool:
    """
    window: (SEQ_LEN, 88), raw velocities 0-127
    """
    if not SKIP_EMPTY_WINDOWS:
        return True

    active_frames = (window > 0).any(axis=1).sum()
    coverage = active_frames / SEQ_LEN

    if coverage < MIN_SEQ_COVERAGE:
        return False

    note_ons = (window > 0).sum()
    if note_ons < MIN_ACTIVE_NOTES:
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Process one MIDI file → list of windows
# ─────────────────────────────────────────────────────────────
def process_file(path: str):
    """
    Parse one MIDI file and extract multiple fixed-length windows
    using sliding window with STRIDE.

    Returns:
        list[np.ndarray]: each item shape (SEQ_LEN, FEATURE_SIZE)
                          raw velocities 0-127, NOT normalised.
                          Normalisation happens in dataset.py at load time.
    """
    notes = parse_midi(path)
    if not notes:
        return []

    max_time     = max(end for _, end, _, _ in notes)
    total_frames = int(max_time * MIDI_FS) + 1

    # Skip files shorter than one window
    if total_frames < SEQ_LEN:
        return []

    full_roll = notes_to_piano_roll(notes, max_len=total_frames, fs=MIDI_FS)
    # full_roll: (total_frames, 88)

    windows = []
    start   = 0
    while start + SEQ_LEN <= total_frames:
        window = full_roll[start : start + SEQ_LEN]   # (SEQ_LEN, 88)

        if passes_quality_filter(window):
            # ── Save as (SEQ_LEN, FEATURE_SIZE) — no flattening ──
            # Cleaner shape; dataset.py does not need to reshape.
            windows.append(window.copy())

        start += STRIDE

    return windows


# ─────────────────────────────────────────────────────────────
# File collection
# ─────────────────────────────────────────────────────────────
def collect_midi_files(raw_dir: str):
    midi_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))
    midi_files.sort()
    return midi_files


# ─────────────────────────────────────────────────────────────
# File-level split  (prevents leakage)
# ─────────────────────────────────────────────────────────────
def split_files(midi_files):
    """
    Split at FILE level BEFORE generating any windows.
    This guarantees no MIDI file's windows appear in more than one split,
    preventing evaluation leakage.
    """
    train_files, temp_files = train_test_split(
        midi_files,
        test_size  = (1.0 - TRAIN_RATIO),
        random_state = SEED,
    )
    val_fraction_in_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files,
        test_size    = (1.0 - val_fraction_in_temp),
        random_state = SEED,
    )
    return train_files, val_files, test_files


# ─────────────────────────────────────────────────────────────
# Build one split
# ─────────────────────────────────────────────────────────────
def build_split_dataset(file_list, split_name="train"):
    """
    Process a list of MIDI files into a 3-D dataset array.

    Returns:
        np.ndarray shape (N, SEQ_LEN, FEATURE_SIZE)  — raw velocities 0-127
    """
    data          = []
    skipped_files = 0

    for path in tqdm(file_list, desc=f"Processing {split_name}"):
        try:
            windows = process_file(path)
            data.extend(windows)
        except Exception as e:
            print(f"  Error: {path} — {e}")
            skipped_files += 1

    if not data:
        print(f"  WARNING: No usable windows found for {split_name}.")
        return np.empty((0, SEQ_LEN, FEATURE_SIZE), dtype=np.float16), skipped_files

    # Stack into (N, SEQ_LEN, FEATURE_SIZE)
    arr = np.stack(data, axis=0).astype(np.float16)
    return arr, skipped_files


# ─────────────────────────────────────────────────────────────
# Metadata snapshot
# ─────────────────────────────────────────────────────────────
def save_metadata(train_files, val_files, test_files,
                  train_data, val_data, test_data):
    metadata = {
        "dataset_name"       : DATASET_NAME,
        "dataset_subset"     : DATASET_SUBSET,
        "representation"     : REPRESENTATION,
        "seq_len"            : SEQ_LEN,
        "feature_size"       : FEATURE_SIZE,
        "midi_fs"            : MIDI_FS,
        "stride"             : STRIDE,
        "min_active_notes"   : MIN_ACTIVE_NOTES,
        "min_seq_coverage"   : MIN_SEQ_COVERAGE,
        "skip_empty_windows" : SKIP_EMPTY_WINDOWS,
        "train_ratio"        : TRAIN_RATIO,
        "val_ratio"          : VAL_RATIO,
        "test_ratio"         : TEST_RATIO,
        "seed"               : SEED,
        "save_shape"         : "(N, SEQ_LEN, FEATURE_SIZE)",
        "normalisation"      : "raw velocities 0-127; divide by 127 in dataset.py",
        "num_train_files"    : len(train_files),
        "num_val_files"      : len(val_files),
        "num_test_files"     : len(test_files),
        "train_shape"        : list(train_data.shape),
        "val_shape"          : list(val_data.shape),
        "test_shape"         : list(test_data.shape),
    }
    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json saved → {meta_path}")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def main():
    print("\nStarting preprocessing pipeline...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_DIR):
        print(f"ERROR: RAW_DIR does not exist: {RAW_DIR}")
        return

    midi_files = collect_midi_files(RAW_DIR)
    print(f"\nMIDI files found : {len(midi_files)}")
    if not midi_files:
        print("ERROR: No MIDI files found.")
        return

    # ── 1. File-level split ────────────────────────────────────
    train_files, val_files, test_files = split_files(midi_files)
    print(f"\nFile-level split ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}):")
    print(f"  Train files : {len(train_files)}")
    print(f"  Val files   : {len(val_files)}")
    print(f"  Test files  : {len(test_files)}")

    # ── 2. Window extraction per split ────────────────────────
    train_data, ts = build_split_dataset(train_files, "train")
    val_data,   vs = build_split_dataset(val_files,   "val")
    test_data,  es = build_split_dataset(test_files,  "test")

    print(f"\nWindow datasets (N, SEQ_LEN, FEATURE_SIZE):")
    print(f"  Train : {train_data.shape}")
    print(f"  Val   : {val_data.shape}")
    print(f"  Test  : {test_data.shape}")
    print(f"  Value range: [{train_data.min():.0f}, {train_data.max():.0f}]  (raw — not yet normalised)")
    print(f"  Files with errors: {ts + vs + es}")

    # ── 3. Save ───────────────────────────────────────────────
    np.save(os.path.join(PROCESSED_DIR, "train.npy"), train_data)
    np.save(os.path.join(PROCESSED_DIR, "val.npy"),   val_data)
    np.save(os.path.join(PROCESSED_DIR, "test.npy"),  test_data)
    save_metadata(train_files, val_files, test_files,
                  train_data, val_data, test_data)

    print(f"\nPreprocessing COMPLETE — saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
