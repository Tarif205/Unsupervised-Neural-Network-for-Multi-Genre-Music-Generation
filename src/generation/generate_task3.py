# src/generation/generate_task3.py
"""
Task 3 — Generate 10 long-sequence MIDI compositions
Faculty guide:
- "Start with seed token (e.g. Bar)"
- "Autoregressively sample next token"
- "Use temperature sampling with T in [0.8, 1.2]"
- "Use top-k sampling with k in [5, 50]"
- "Convert token sequence to MIDI using miditok"
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR     = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from miditok import REMI
from config import (
    DEVICE, MODEL_DIR, MIDI_DIR, PROCESSED_DIR,
    NHEAD, NUM_ENC_LAYERS, DIM_FEEDFORWARD, TF_DROPOUT, TF_MAX_LEN
)
from models.transformer import MusicTransformer


def load_tokenizer():
    tokenizer_path = os.path.join(PROCESSED_DIR, 'remi_tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f'Tokenizer not found: {tokenizer_path}')
    tokenizer = REMI(params=Path(tokenizer_path))
    print(f'✅ Tokenizer loaded | vocab={len(tokenizer.vocab)}')
    return tokenizer


def load_model(vocab_size):
    model = MusicTransformer(
        vocab_size = vocab_size,
        d_model    = 256,
        n_heads    = NHEAD,
        n_layers   = NUM_ENC_LAYERS,
        d_ff       = DIM_FEEDFORWARD,
        max_len    = TF_MAX_LEN - 1,
        dropout    = TF_DROPOUT,
    ).to(DEVICE)

    ckpt = os.path.join(MODEL_DIR, 'task3_transformer_best.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Model not found: {ckpt}')

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'✅ Model loaded: {ckpt}')
    return model


def get_seed_token(tokenizer):
    """Get Bar token ID as seed — faculty recommendation."""
    vocab = tokenizer.vocab
    # Try different Bar token names in REMI
    for name in ['Bar_None', 'Bar', 'BAR']:
        if name in vocab:
            return vocab[name]
    # Fallback to first token
    return 1


# def generate_composition(
#     model,
#     tokenizer,
#     out_path,
#     n_tokens    = 512,
#     temperature = 1.0,
#     top_k       = 50,
# ):
#     """Generate one composition and save as MIDI."""
#     seed_id = get_seed_token(tokenizer)
#     seed    = torch.tensor([[seed_id]], dtype=torch.long, device=DEVICE)

#     with torch.no_grad():
#         generated = model.generate(
#             seed_tokens    = seed,
#             max_new_tokens = n_tokens,
#             temperature    = temperature,
#             top_k          = top_k,
#         )

#     token_ids = generated[0].cpu().tolist()

#     try:
#         # New miditok 3.x API — use decode() instead of tokens_to_midi()
#         score = tokenizer.decode([token_ids])

#         # score is a symusic.Score object — save directly
#         score.dump_midi(out_path)

#         # Fix drum instrument using pretty_midi after saving
#         import pretty_midi
#         midi = pretty_midi.PrettyMIDI(out_path)
#         for instrument in midi.instruments:
#             instrument.is_drum = True
#             instrument.program = 0
#         midi.write(out_path)

#         return True, len(token_ids)

#     except Exception as e:
#         print(f'  ⚠️  MIDI conversion failed: {e}')
#         return False, 0

def generate_composition(
    model,
    tokenizer,
    out_path,
    n_tokens    = 512,
    temperature = 1.0,
    top_k       = 50,
):
    """Generate one composition and save as MIDI."""
    seed_id = get_seed_token(tokenizer)
    seed    = torch.tensor([[seed_id]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = model.generate(
            seed_tokens    = seed,
            max_new_tokens = n_tokens,
            temperature    = temperature,
            top_k          = top_k,
        )

    token_ids = generated[0].cpu().tolist()

    try:
        import pretty_midi
        import tempfile

        # Decode tokens to symusic Score
        score = tokenizer.decode([token_ids])

        # ── Windows-safe export ────────────────────────────────
        # Write to temp file first, then copy to destination
        # This avoids Windows file locking issues with symusic
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mid')
        os.close(tmp_fd)

        try:
            score.dump_midi(tmp_path)
        except Exception:
            # Fallback: use pretty_midi to build MIDI from scratch
            return _fallback_export(score, out_path, token_ids)

        # Load with pretty_midi to fix drum instrument
        midi = pretty_midi.PrettyMIDI(tmp_path)
        for instrument in midi.instruments:
            instrument.is_drum = True
            instrument.program = 0
        midi.write(out_path)

        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass

        return True, len(token_ids)

    except Exception as e:
        print(f'  ⚠️  MIDI conversion failed: {e}')
        return False, 0


def _fallback_export(score, out_path, token_ids):
    """
    Fallback: manually build pretty_midi object from symusic score.
    Used when symusic dump_midi fails on Windows.
    """
    import pretty_midi
    try:
        midi       = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        instrument = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        # Extract notes from symusic score tracks
        for track in score.tracks:
            for note in track.notes:
                # symusic uses ticks — convert to seconds
                ticks_per_beat = score.ticks_per_quarter
                tempo          = 500000  # default 120 BPM in microseconds
                start_sec      = note.time   / ticks_per_beat * (tempo / 1e6)
                end_sec        = (note.time + note.duration) / ticks_per_beat * (tempo / 1e6)
                velocity       = max(1, min(127, note.velocity))

                pm_note = pretty_midi.Note(
                    velocity = velocity,
                    pitch    = note.pitch,
                    start    = start_sec,
                    end      = max(start_sec + 0.05, end_sec),
                )
                instrument.notes.append(pm_note)

        if instrument.notes:
            midi.instruments.append(instrument)
            midi.write(out_path)
            m     = pretty_midi.PrettyMIDI(out_path)
            notes = sum(len(i.notes) for i in m.instruments)
            return True, len(token_ids)
        else:
            return False, 0

    except Exception as e:
        print(f'  ⚠️  Fallback export failed: {e}')
        return False, 0


def main():
    out_dir = os.path.join(MIDI_DIR, 'task3')
    os.makedirs(out_dir, exist_ok=True)

    tokenizer  = load_tokenizer()
    vocab_size = len(tokenizer.vocab)
    model      = load_model(vocab_size)

    print(f'\nGenerating 10 compositions...\n')

    # Faculty: temperature in [0.8, 1.2], top-k in [5, 50]
    configs = [
        {'temperature': 0.8,  'top_k': 50,  'label': 'conservative'},
        {'temperature': 0.9,  'top_k': 50,  'label': 'balanced'},
        {'temperature': 1.0,  'top_k': 50,  'label': 'standard'},
        {'temperature': 1.0,  'top_k': 20,  'label': 'focused'},
        {'temperature': 1.0,  'top_k': 10,  'label': 'tight'},
        {'temperature': 1.1,  'top_k': 50,  'label': 'creative'},
        {'temperature': 1.2,  'top_k': 50,  'label': 'diverse'},
        {'temperature': 0.9,  'top_k': 30,  'label': 'smooth'},
        {'temperature': 1.0,  'top_k': 40,  'label': 'natural'},
        {'temperature': 1.1,  'top_k': 25,  'label': 'expressive'},
    ]

    saved = 0
    for i, cfg in enumerate(configs):
        out_path = os.path.join(out_dir, f'task3_sample_{i+1:02d}_{cfg["label"]}.mid')
        success, n_tokens = generate_composition(
            model, tokenizer, out_path,
            n_tokens    = 512,
            temperature = cfg['temperature'],
            top_k       = cfg['top_k'],
        )

        if success:
            # Verify with pretty_midi
            try:
                import pretty_midi
                m     = pretty_midi.PrettyMIDI(out_path)
                notes = sum(len(inst.notes) for inst in m.instruments)
                dur   = m.get_end_time()
                print(f'  [{i+1}/10] ✅ {os.path.basename(out_path)} | '
                      f'tokens={n_tokens} | notes={notes} | duration={dur:.1f}s | '
                      f'T={cfg["temperature"]} top_k={cfg["top_k"]}')
                saved += 1
            except Exception as e:
                print(f'  [{i+1}/10] ⚠️  Generated but unreadable: {e}')
        else:
            print(f'  [{i+1}/10] ❌ Failed')

    print(f'\n✅ {saved}/10 compositions saved to {out_dir}')


if __name__ == '__main__':
    main()