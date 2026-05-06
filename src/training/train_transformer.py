# src/training/train_transformer.py
"""
Task 3 — Autoregressive Transformer Training
Faculty guide:
- Input: token sequence [x1, ..., x_{T-1}]
- Target: same sequence shifted one position [x2, ..., x_T]
- Loss: cross-entropy averaged across all positions
- Monitor: perplexity on validation set each epoch
- Perplexity = exp(average cross-entropy per token)
"""

import os
import sys
import math
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR     = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    DEVICE, MODEL_DIR, PLOT_DIR, PROCESSED_DIR,
    BATCH_SIZE, EPOCHS, LR, CLIP_GRAD, SEED,
    NHEAD, NUM_ENC_LAYERS, DIM_FEEDFORWARD, TF_DROPOUT, TF_MAX_LEN
)
from models.transformer import MusicTransformer


# ── Token Dataset ──────────────────────────────────────────────
class TokenDataset(Dataset):
    """
    Loads tokenized MIDI sequences from .npy file.
    Returns (input, target) pairs for autoregressive training:
      input  = seq[:-1]  (all tokens except last)
      target = seq[1:]   (all tokens except first)
    """
    def __init__(self, split='train'):
        token_dir = os.path.join(PROCESSED_DIR, 'tokens')
        path      = os.path.join(token_dir, f'{split}_tokens.npy')

        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Token file not found: {path}\n'
                f'Run: python src/preprocessing/tokenize_midi.py'
            )

        self.data = np.load(path, mmap_mode='r')
        print(f'[{split}] {len(self.data):,} sequences | shape {self.data.shape}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq    = torch.tensor(self.data[idx].copy(), dtype=torch.long)
        input_ = seq[:-1]   # x_1 ... x_{T-1}
        target = seq[1:]    # x_2 ... x_T
        return input_, target


def load_vocab_size():
    """Load vocab size from tokenizer metadata."""
    meta_path = os.path.join(PROCESSED_DIR, 'tokens', 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)['vocab_size']
    raise FileNotFoundError(f'metadata.json not found at {meta_path}')


def compute_perplexity(loss):
    """Perplexity = exp(average cross-entropy per token)"""
    return math.exp(min(loss, 20))   # cap at exp(20) to avoid overflow


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_tokens = 0

    for input_, target in loader:
        input_  = input_.to(DEVICE)
        target  = target.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        logits = model(input_)  # (B, T-1, vocab)

        # Reshape for cross-entropy: (B*(T-1), vocab) vs (B*(T-1),)
        B, T, V  = logits.shape
        loss     = criterion(logits.reshape(B*T, V), target.reshape(B*T))

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        total_loss   += loss.item() * B * T
        total_tokens += B * T

    avg_loss = total_loss / total_tokens
    return avg_loss


def main():
    torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    # Load data
    train_dataset = TokenDataset('train')
    val_dataset   = TokenDataset('val')

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    print(f'Train sequences: {len(train_dataset):,}')
    print(f'Val   sequences: {len(val_dataset):,}')

    # Build model
    vocab_size = load_vocab_size()
    model = MusicTransformer(
        vocab_size = vocab_size,
        d_model    = 256,
        n_heads    = NHEAD,
        n_layers   = NUM_ENC_LAYERS,
        d_ff       = DIM_FEEDFORWARD,
        max_len    = TF_MAX_LEN - 1,  # -1 because input = seq[:-1]
        dropout    = TF_DROPOUT,
    ).to(DEVICE)

    # Ignore padding token (id=0) in loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler — cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_val_loss = float('inf')
    train_losses, val_losses   = [], []
    train_perps,  val_perps    = [], []

    best_path  = os.path.join(MODEL_DIR, 'task3_transformer_best.pt')
    final_path = os.path.join(MODEL_DIR, 'task3_transformer_final.pt')

    print(f'\nStarting training for {EPOCHS} epochs...\n')

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        tr_ppl = compute_perplexity(tr_loss)
        vl_ppl = compute_perplexity(vl_loss)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_perps.append(tr_ppl)
        val_perps.append(vl_ppl)

        print(
            f'Epoch {epoch:02d}/{EPOCHS} | '
            f'Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | '
            f'Train PPL: {tr_ppl:.2f} | Val PPL: {vl_ppl:.2f} | '
            f'LR: {scheduler.get_last_lr()[0]:.6f}'
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), best_path)
            print(f'  ✅ Saved best model (val_ppl={vl_ppl:.2f})')

    torch.save(model.state_dict(), final_path)

    # ── Plots ──────────────────────────────────────────────────
    epochs_r = list(range(1, EPOCHS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Task 3 — Transformer Training Curves', fontsize=13, fontweight='bold')

    axes[0].plot(epochs_r, train_losses, label='Train Loss', color='steelblue', linewidth=2)
    axes[0].plot(epochs_r, val_losses,   label='Val Loss',   color='coral',
                 linewidth=2, linestyle='--')
    axes[0].set_title('Cross-Entropy Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_r, train_perps, label='Train PPL', color='steelblue', linewidth=2)
    axes[1].plot(epochs_r, val_perps,   label='Val PPL',   color='coral',
                 linewidth=2, linestyle='--')
    axes[1].set_title('Perplexity (lower = better)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'task3_training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f'\n✅ Training complete!')
    print(f'   Best Val Loss      : {best_val_loss:.4f}')
    print(f'   Best Val Perplexity: {compute_perplexity(best_val_loss):.2f}')
    print(f'   Plot saved → {plot_path}')


if __name__ == '__main__':
    main()