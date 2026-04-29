# src/generation/generate_music.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEVICE, LATENT_DIM, HIDDEN_DIM, FEATURE_SIZE,
    NUM_LAYERS, DROPOUT, MIDI_DIR, MODEL_DIR, PLOT_DIR,
    MIDI_FS, MIDI_THRESHOLD
)
from models.autoencoder import LSTMAutoencoder
from generation.midi_export import piano_roll_to_midi


def generate_samples(
    n         : int   = 5,
    noise_std : float = 1.0,
    task_tag  : str   = "task1",
    ckpt_path : str   = None,
    seq_len   : int   = 500,
):
    if ckpt_path is None:
        ckpt_path = os.path.join(MODEL_DIR, "task1_lstm_autoencoder_best.pt")

    out_dir = os.path.join(MIDI_DIR, task_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = LSTMAutoencoder(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path}")

    paths = []
    with torch.no_grad():
        for i in range(n):
            # Sample from prior N(0, I)
            z = torch.randn(1, LATENT_DIM, device=DEVICE) * noise_std

            # Decode latent to piano roll
            roll_tensor = model.decode(z, seq_len=seq_len)
            roll_np     = roll_tensor.squeeze(0).cpu().numpy()  # (T, 88)

            # Save MIDI
            path = os.path.join(out_dir, f"{task_tag}_sample_{i+1:02d}.mid")
            piano_roll_to_midi(
                piano_roll  = roll_np,
                output_path = path,
                fs          = MIDI_FS,
                threshold   = MIDI_THRESHOLD,
            )
            paths.append(path)
            print(f"  [{i+1}/{n}] Saved → {path}")

    print(f"\n🎵 {n} MIDI files saved to {out_dir}")
    return paths


def visualise_piano_roll(roll_np: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(
        roll_np.T,
        aspect = "auto",
        origin = "lower",
        cmap   = "hot",
        vmin   = 0,
        vmax   = 1,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Pitch (index 0 = A0)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Piano-roll image saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=5)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--ckpt",      type=str,   default=None)
    args = parser.parse_args()

    paths = generate_samples(
        n         = args.n,
        noise_std = args.noise_std,
        ckpt_path = args.ckpt,
    )