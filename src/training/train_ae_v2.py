# src/training/train_ae_v2.py
import os, sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import *
from dataset import GrooveDataset
from models.autoencoder import LSTMAutoencoder


def weighted_mse_loss(output, target, pos_weight=20.0):
    """
    Penalize missing notes 20x more than predicting silence.
    This fixes the sparsity problem.
    """
    weights = torch.where(target > 0.15,
                          torch.tensor(pos_weight, device=target.device),
                          torch.tensor(1.0, device=target.device))
    loss = weights * (output - target) ** 2
    return loss.mean()

# def weighted_mse_loss(output, target, pos_weight=20.0):
#     pos_weight = torch.tensor(pos_weight, device=target.device, dtype=target.dtype)
#     weights = torch.where(target > 0.15, pos_weight, 1.0)
#     return (weights * (output - target) ** 2).mean()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_samples = 0.0, 0

    for batch in loader:
        batch = batch.to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        output = model(batch)
        loss = weighted_mse_loss(output, batch)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        total_loss    += loss.item() * batch.size(0)
        total_samples += batch.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def main():
    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    train_dataset = GrooveDataset(split='train')
    val_dataset   = GrooveDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train samples: {len(train_dataset):,}')
    print(f'Val   samples: {len(val_dataset):,}')

    model = LSTMAutoencoder(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    best_model_path = os.path.join(MODEL_DIR, 'task1_lstm_autoencoder_best_v2.pt')
    plot_path       = os.path.join(PLOT_DIR,  'task1_loss_curve_v2.png')

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss   = run_epoch(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'  ✅ Saved best model')

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.title('Task 1 LSTM Autoencoder — Weighted Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f'📊 Loss plot saved → {plot_path}')


if __name__ == '__main__':
    main()