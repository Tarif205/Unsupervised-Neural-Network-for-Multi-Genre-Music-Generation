# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(__file__))
from config import PROCESSED_DIR, SEQ_LEN, FEATURE_SIZE


class GrooveDataset(Dataset):
    """
    Memory-safe dataset for large .npy piano-roll files.

    Key design decisions:
    ─────────────────────────────────────────────────────────────
    1. mmap_mode='r'   — file is NEVER fully loaded into RAM.
                         Only the requested batch pages are read
                         from disk on demand. Safe for 15+ GB files
                         on 8 GB RAM.

    2. Normalisation   — done per sample in __getitem__, NOT on the
                         full array. Doing (raw / 127.0) on an mmap
                         array would materialise the entire file into
                         RAM, defeating mmap entirely.

    3. .copy()         — mmap returns a read-only view. .copy() pulls
                         exactly one sample (500×88 floats = ~176 KB)
                         into RAM as a writeable array before converting
                         to float32 and normalising.

    4. num_workers=0   — use this in DataLoader on Windows. Multiple
                         worker processes each hold mmap handles and
                         can cause RAM spikes or crashes on Windows.

    Saved shape expected : (N, SEQ_LEN, FEATURE_SIZE)  →  (N, 500, 88)
    Legacy flat shape    : (N, SEQ_LEN * FEATURE_SIZE) →  (N, 44000)
                           handled automatically.
    """

    def __init__(self, split="train", seq_len=SEQ_LEN, feature_size=FEATURE_SIZE):
        path = os.path.join(PROCESSED_DIR, f"{split}.npy")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")

        # ── Memory-mapped load — does NOT read file into RAM ─────
        raw = np.load(path, mmap_mode='r')

        # ── Handle legacy flat shape (N, 44000) ──────────────────
        if raw.ndim == 2:
            assert raw.shape[1] == seq_len * feature_size, (
                f"Flat shape mismatch: expected {seq_len * feature_size}, "
                f"got {raw.shape[1]}"
            )
            raw = raw.reshape(len(raw), seq_len, feature_size)

        assert raw.ndim == 3, f"Expected 3-D array, got shape {raw.shape}"
        assert raw.shape[1] == seq_len,      f"SEQ_LEN mismatch:     {raw.shape[1]} vs {seq_len}"
        assert raw.shape[2] == feature_size, f"FEATURE_SIZE mismatch: {raw.shape[2]} vs {feature_size}"

        # Store mmap reference — no data in RAM yet
        self.data         = raw
        self.seq_len      = seq_len
        self.feature_size = feature_size
        self.split        = split

        file_size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"[{split}] {len(raw):,} samples | "
              f"shape {raw.shape} | "
              f"file {file_size_gb:.2f} GB | "
              f"mmap — RAM usage: minimal")

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     # .copy() fetches only this one sample from disk (~176 KB)
    #     # then immediately converts and normalises in RAM
    #     x = self.data[idx].copy().astype(np.float32)   # (500, 88)
    #     x = x / 127.0                                   # → [0.0, 1.0]
    #     # return torch.tensor(x)                          # (SEQ_LEN, FEATURE_SIZE)
    #     return torch.from_numpy(x)

    def __getitem__(self, idx):
    # load one sample
        x = self.data[idx].copy().astype(np.float32)

        # normalize to [0,1]
        x = x / 127.0

        # convert to binary piano roll
        x = (x > 0).astype(np.float32)

        return torch.from_numpy(x)