# src/generation/generate_task2_samples.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pretty_midi
from config import (
    MODEL_DIR, MIDI_DIR, DEVICE, SEQ_LEN,
    FEATURE_SIZE, HIDDEN_DIM, LATENT_DIM,
    NUM_LAYERS, DROPOUT, MIDI_FS, MIDI_THRESHOLD
)
from models.vae import MusicVAE
from generation.midi_export import piano_roll_to_midi


def main():
    out_dir = os.path.join(MIDI_DIR, 'task2')
    os.makedirs(out_dir, exist_ok=True)

    model = MusicVAE(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    model_path = os.path.join(MODEL_DIR, 'best_vae.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'VAE model not found: {model_path}')

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'✅ Loaded VAE model: {model_path}')

    with torch.no_grad():
        samples = model.sample(num_samples=8, seq_len=SEQ_LEN, device=DEVICE)

    samples = samples.cpu().numpy()

    for i in range(samples.shape[0]):
        roll = samples[i]
        path = os.path.join(out_dir, f'task2_sample_{i+1:02d}.mid')
        piano_roll_to_midi(
            piano_roll  = roll,
            output_path = path,
            fs          = MIDI_FS,
            threshold   = 0.10,
        )
        m     = pretty_midi.PrettyMIDI(path)
        notes = sum(len(inst.notes) for inst in m.instruments)
        dur   = m.get_end_time()
        print(f'  [{i+1}/8] Notes:{notes} | Duration:{dur:.1f}s | {os.path.basename(path)}')

    print(f'\n✅ 8 MIDI samples saved to {out_dir}')


if __name__ == '__main__':
    main()