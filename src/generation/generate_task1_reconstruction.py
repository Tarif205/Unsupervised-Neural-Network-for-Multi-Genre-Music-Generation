
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pretty_midi

from config import (
    DEVICE, FEATURE_SIZE, HIDDEN_DIM, LATENT_DIM,
    NUM_LAYERS, DROPOUT, MIDI_DIR, MODEL_DIR, MIDI_FS
)
from dataset import GrooveDataset
from models.autoencoder import LSTMAutoencoder
from generation.midi_export import piano_roll_to_midi


def main():
    # ── Load model ───────────────────────────
    model = LSTMAutoencoder(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    ckpt = os.path.join(MODEL_DIR, "task1_ae_v3_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f'✅ Loaded: {ckpt}')

    # ── Load dataset ─────────────────────────
    dataset = GrooveDataset(split='test')

    out_dir = os.path.join(MIDI_DIR, 'task1')
    os.makedirs(out_dir, exist_ok=True)

    saved = 0

    # 🔥 Pick RANDOM DISTINCT samples manually
    indices = torch.randperm(len(dataset))

    with torch.no_grad():
        for idx in indices:

            batch = dataset[idx].unsqueeze(0)

            # skip weak samples
            if batch.sum() < 50:
                continue

            batch = batch.to(DEVICE)

            # 🔥 RANDOM NOISE (different each sample)
            noise_level = np.random.uniform(0.05, 0.15)
            noisy_batch = batch + noise_level * torch.randn_like(batch)

            recon = model(noisy_batch)

            # sigmoid for BCE
            recon = torch.sigmoid(recon)

            roll = recon[0].cpu().numpy()

            # 🔥 ADD SMALL RANDOMNESS (ensures variation)
            roll = roll + 0.03 * np.random.randn(*roll.shape)
            roll = np.clip(roll, 0, 1)

            print(f'Sample {saved+1} | noise:{noise_level:.3f} | min:{roll.min():.4f} max:{roll.max():.4f} mean:{roll.mean():.4f}')

            path = os.path.join(out_dir, f'task1_sample_{saved+1:02d}.mid')

            piano_roll_to_midi(
                piano_roll  = roll,
                output_path = path,
                fs          = MIDI_FS,
                threshold   = 0.05,
            )

            # ── Validate ───────────────────────
            m = pretty_midi.PrettyMIDI(path)
            notes = sum(len(i.notes) for i in m.instruments)
            duration = m.get_end_time()

            print(f'  Notes: {notes} | Duration: {duration:.1f}s')

            # keep only valid samples
            if notes >= 50 and duration >= 5:
                saved += 1
            else:
                os.remove(path)
                continue

            if saved >= 5:
                break

    print(f'\n✅ Done! {saved} VALID samples saved to {out_dir}')


if __name__ == '__main__':
    main()