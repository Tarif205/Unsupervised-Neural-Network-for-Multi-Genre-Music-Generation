# src/training/train_ae_v3.py
import os, sys, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR     = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    MODEL_DIR, PLOT_DIR, DEVICE, FEATURE_SIZE,
    HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, CLIP_GRAD,
)
from dataset import GrooveDataset
from models.autoencoder import LSTMAutoencoder, FocalLoss


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_items = 0

    for batch in loader:
        x = batch.to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss   = criterion(logits, x)  # x is target (binarized)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        total_loss  += loss.item() * x.size(0)
        total_items += x.size(0)

    return total_loss / total_items


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    train_dataset = GrooveDataset(split='train')
    val_dataset   = GrooveDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train: {len(train_dataset):,} | Val: {len(val_dataset):,}')

    model = LSTMAutoencoder(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    # Faculty recommended: Focal Loss
    criterion = FocalLoss(gamma=2.0, pos_weight=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val   = float('inf')
    train_hist = []
    val_hist   = []
    best_path  = os.path.join(MODEL_DIR, 'task1_ae_v3_best.pt')

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        vl = run_epoch(model, val_loader,   criterion)

        train_hist.append(tr)
        val_hist.append(vl)

        print(f'Epoch {epoch:02d}/{EPOCHS} | Train: {tr:.6f} | Val: {vl:.6f}')

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
            print(f'  ✅ Saved best model')

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.title('Task 1 AE v3 — Focal Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'task1_ae_v3_loss.png'), dpi=150)
    plt.close()
    print('✅ Training complete!')


if __name__ == '__main__':
    main()