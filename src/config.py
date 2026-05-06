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
# DATASET_NAME   = "maestro"
# DATASET_SUBSET = "all"
# GENRES         = ["piano"]

# ── Dataset identity ──────────────────────────────────────────
DATASET_NAME    = "groove_midi"          # groove_midi | maestro | lakh
DATASET_SUBSET  = "drummer1/session1"    # subfolder used; "all" if full dataset
GENRES          = ["drums", "rhythm"]    # genres present in this dataset

# ── Representation ────────────────────────────────────────────
REPRESENTATION = "piano_roll"

# ── Preprocessing / windowing ─────────────────────────────────
# Faculty guide: "At fs=16, 128 time steps = 8 seconds of music"
# We use fs=100 (10ms resolution) with 128 frames = 1.28 seconds
# This gives finer timing resolution for piano music
SEQ_LEN      = 128   # faculty minimum — covers ~1.28s at fs=100
FEATURE_SIZE = 88    # piano keys MIDI 21 (A0) to 108 (C8)
MIDI_FS      = 16   # frames per second (10ms resolution)

# Non-overlapping windows — no data leakage
# Set to SEQ_LEN//2 for 50% overlap (more samples)
STRIDE       = 128   # = SEQ_LEN for non-overlapping windows

SAVE_DTYPE   = "float32"

# Quality filters
MIN_ACTIVE_NOTES   = 30     # minimum note-on events per window
MIN_SEQ_COVERAGE   = 0.05   # minimum fraction of active frames (5%)
SKIP_EMPTY_WINDOWS = True

# ── Train / val / test split ──────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Model (Task 1 & 2 shared LSTM backbone) ───────────────────
# Faculty: "2-layer LSTM, hidden 256, latent 64 is reasonable"
HIDDEN_DIM = 256
LATENT_DIM = 64     # faculty says 64 or 128
NUM_LAYERS = 2
DROPOUT    = 0.3

# ── Training ──────────────────────────────────────────────────
# Faculty: "batch size 64, reduce to 32 if GPU memory insufficient"
# RTX 3050 4GB → use 32
BATCH_SIZE = 32
EPOCHS     = 30     # more epochs needed with focal loss
LR         = 1e-3   # faculty: "Adam with lr=1e-3 as starting point"
CLIP_GRAD  = 1.0
SEED       = 42

# ── VAE (Task 2) ──────────────────────────────────────────────
BETA         = 1.0   # final KL weight after annealing
KL_WARMUP    = 10    # epochs with β=0 before annealing starts

# ── Transformer (Task 3) ──────────────────────────────────────
NHEAD           = 4
NUM_ENC_LAYERS  = 4
DIM_FEEDFORWARD = 512
TF_DROPOUT      = 0.1
TF_MAX_LEN      = 512

# ── Generation ────────────────────────────────────────────────
# Faculty: "lower threshold below 0.5 since model underestimates notes"
MIDI_THRESHOLD = 0.15
MIDI_TEMPO     = 120.0

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Sanity checks ─────────────────────────────────────────────
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"
assert STRIDE > 0, "STRIDE must be positive"
assert REPRESENTATION in ("piano_roll", "token"), \
    "REPRESENTATION must be 'piano_roll' or 'token'"