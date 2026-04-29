# src/generation/generate_task1_reconstruction.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pretty_midi
from torch.utils.data import DataLoader

from config import (
    DEVICE, FEATURE_SIZE, HIDDEN_DIM, LATENT_DIM,
    NUM_LAYERS, DROPOUT, MIDI_DIR, MODEL_DIR, MIDI_FS
)
from dataset import GrooveDataset
from models.autoencoder import LSTMAutoencoder
from generation.midi_export import piano_roll_to_midi


def main():
    # Load model
    model = LSTMAutoencoder(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    ckpt = os.path.join(MODEL_DIR, 'task1_lstm_autoencoder_best_v2.pt')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'✅ Loaded: {ckpt}')

    # Load test set
    dataset = GrooveDataset(split='test')
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    out_dir = os.path.join(MIDI_DIR, 'task1_v2')
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            if batch.sum() < 50:
                continue

            batch = batch.to(DEVICE)
            recon = model(batch)
            roll  = recon[0].cpu().numpy()

            print(f'Sample {saved+1} - min:{roll.min():.4f} max:{roll.max():.4f} mean:{roll.mean():.6f}')

            path = os.path.join(out_dir, f'task1_v2_sample_{saved+1:02d}.mid')
            piano_roll_to_midi(
                piano_roll  = roll,
                output_path = path,
                fs          = MIDI_FS,
                threshold   = 0.05,   # was 0.10
            )

            # Verify notes
            m      = pretty_midi.PrettyMIDI(path)
            notes  = sum(len(i.notes) for i in m.instruments)
            duration = m.get_end_time()
            print(f'  Notes: {notes} | Duration: {duration:.1f}s | Saved: {os.path.basename(path)}')

            saved += 1
            if saved >= 5:
                break

    print(f'\n✅ Done! {saved} samples saved to {out_dir}')


if __name__ == '__main__':
    main()