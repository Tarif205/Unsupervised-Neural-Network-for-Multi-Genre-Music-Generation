# src/config.py
# ─────────────────────────────────────────────────────────────
# Central config — edit here, imports everywhere else stay clean
# ─────────────────────────────────────────────────────────────
import os
import torch

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw_midi")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLIT_DIR     = os.path.join(BASE_DIR, "data", "train_test_split")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "plots")
MIDI_DIR      = os.path.join(OUTPUT_DIR, "generated_midis")

# ── Dataset identity ──────────────────────────────────────────
# DATASET_NAME    = "groove_midi"          # groove_midi | maestro | lakh
# DATASET_SUBSET  = "drummer1/session1"    # subfolder used; "all" if full dataset
# GENRES          = ["drums", "rhythm"]    # genres present in this dataset

DATASET_NAME    = "maestro"          # groove_midi | maestro | lakh
DATASET_SUBSET  = "all"    # subfolder used; "all" if full dataset
GENRES          = ["piano"]    # genres present in this dataset

# ── Representation ────────────────────────────────────────────
# "piano_roll" : 2-D matrix  (T x 88),  values = normalised velocity
# "token"      : 1-D sequence of discrete event tokens
REPRESENTATION  = "piano_roll"

# ── Preprocessing / windowing ─────────────────────────────────
SEQ_LEN         = 300    # frames per window (= 5 seconds at 100 fps)
FEATURE_SIZE    = 88     # piano keys, MIDI 21 (A0) to 108 (C8)

# Time grid: 1 frame = 1/MIDI_FS seconds
# At MIDI_FS=100 -> 10 ms resolution, ~1/16-note at 150 BPM
MIDI_FS         = 100    # frames per second

# Sliding-window stride.
# STRIDE = SEQ_LEN  -> non-overlapping windows (no data leakage)
# STRIDE < SEQ_LEN  -> overlapping windows (more training samples)
STRIDE          = 250   # set to 250 for 50% overlap
SAVE_DTYPE = "float32"  # "float32" for max precision, "float16" to save disk space (may cause issues on some platforms)    
# Quality filters — windows that fail these checks are discarded
# MIN_ACTIVE_NOTES    = 10     # minimum total note-on events in a window
# MIN_SEQ_COVERAGE    = 0.02   # minimum fraction of active frames (2%)
MIN_ACTIVE_NOTES    = 30     # minimum total note-on events in a window
MIN_SEQ_COVERAGE    = 0.05
 
SKIP_EMPTY_WINDOWS  = True   # drop windows that are entirely silent

# ── Train / val / test split ratios ───────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10          # must sum to 1.0

# ── Model (Task 1 & 2 shared LSTM backbone) ───────────────────
HIDDEN_DIM  = 256
LATENT_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
CLIP_GRAD   = 1.0
SEED        = 42

# ── VAE (Task 2) ──────────────────────────────────────────────
BETA        = 1.0    # KL weight; >1 = beta-VAE (more disentangled)

# ── Transformer (Task 3) ──────────────────────────────────────
NHEAD           = 4
NUM_ENC_LAYERS  = 4
DIM_FEEDFORWARD = 512
TF_DROPOUT      = 0.1
TF_MAX_LEN      = 512

# ── Generation ────────────────────────────────────────────────
MIDI_THRESHOLD  = 0.1   # minimum activation to count as note-on
MIDI_TEMPO      = 120.0  # BPM of exported MIDI files

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sanity checks (run on import) ─────────────────────────────
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"
assert STRIDE > 0, "STRIDE must be positive"
assert REPRESENTATION in ("piano_roll", "token"), \
    "REPRESENTATION must be 'piano_roll' or 'token'"
