import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from midi_parser import parse_midi
from piano_roll import notes_to_piano_roll
from tokenizer import tokenize


# ======================
# CONFIG (SAFE PATHS)
# ======================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw_midi")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed")

MAX_LEN = 500
FS = 100


print("🚀 Script loaded successfully")


# ======================
# PROCESS SINGLE FILE
# ======================
def process_file(path):
    notes = parse_midi(path)
    piano_roll = notes_to_piano_roll(notes, max_len=MAX_LEN, fs=FS)
    tokens = tokenize(piano_roll)
    return tokens


# ======================
# LOAD DATASET
# ======================
def load_dataset():
    data = []
    midi_files = []

    print("🔍 Scanning dataset at:", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        print("❌ DATA_PATH does not exist!")
        return np.array([])

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(root, file))

    print(f"🎵 Total MIDI files found: {len(midi_files)}")

    if len(midi_files) == 0:
        print("❌ No MIDI files found. Check dataset location.")
        return np.array([])

    for path in tqdm(midi_files, desc="Processing MIDI"):
        try:
            sample = process_file(path)
            data.append(sample)
        except Exception as e:
            print("❌ Error processing:", path)

    return np.array(data)


# ======================
# MAIN PIPELINE
# ======================
def main():
    print("🚀 Starting preprocessing pipeline...")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    data = load_dataset()

    if len(data) == 0:
        print("❌ No data processed. Exiting.")
        return

    print("\n📊 Dataset shape:", data.shape)

    # Split dataset
    train, temp = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    print("\n📦 Data Split:")
    print("Train:", train.shape)
    print("Val:  ", val.shape)
    print("Test: ", test.shape)

    # Save
    np.save(os.path.join(OUTPUT_PATH, "train.npy"), train)
    np.save(os.path.join(OUTPUT_PATH, "val.npy"), val)
    np.save(os.path.join(OUTPUT_PATH, "test.npy"), test)

    print("\n✅ Preprocessing COMPLETE!")
    print("Saved at:", OUTPUT_PATH)


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    main()
