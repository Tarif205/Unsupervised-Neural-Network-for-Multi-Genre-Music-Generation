#src/training/train_ae.py
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (
    BATCH_SIZE,
    CLIP_GRAD,
    DEVICE,
    EPOCHS,
    FEATURE_SIZE,
    HIDDEN_DIM,
    LATENT_DIM,
    LR,
    MIDI_DIR,
    MODEL_DIR,
    NUM_LAYERS,
    DROPOUT,
    PLOT_DIR,
    SEED,
)
from dataset import GrooveDataset
from models.autoencoder import LSTMAutoencoder


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(split, batch_size, shuffle):
    dataset = GrooveDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(DEVICE)

        if optimizer is not None:
            optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, batch)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0

    return total_loss / total_samples


def save_loss_plot(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Task 1 LSTM Autoencoder Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MIDI_DIR, exist_ok=True)

    train_dataset, train_loader = create_loader('train', BATCH_SIZE, True)
    val_dataset, val_loader = create_loader('val', BATCH_SIZE, False)

    print('Train samples:', len(train_dataset))
    print('Validation samples:', len(val_dataset))

    if len(train_dataset) == 0:
        raise ValueError('Train dataset is empty. Run preprocessing first.')

    model = LSTMAutoencoder(
        input_dim=FEATURE_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    best_model_path = os.path.join(MODEL_DIR, 'task1_lstm_autoencoder_best.pt')
    final_model_path = os.path.join(MODEL_DIR, 'task1_lstm_autoencoder_final.pt')
    plot_path = os.path.join(PLOT_DIR, 'task1_loss_curve.png')

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            'Epoch {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f}'.format(
                epoch + 1, EPOCHS, train_loss, val_loss
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print('Saved best model to:', best_model_path)

    torch.save(model.state_dict(), final_model_path)
    print('Saved final model to:', final_model_path)

    save_loss_plot(train_losses, val_losses, plot_path)
    print('Saved loss plot to:', plot_path)


if __name__ == '__main__':
    main()
