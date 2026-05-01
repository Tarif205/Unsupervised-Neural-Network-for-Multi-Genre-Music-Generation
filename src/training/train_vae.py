# Change this:
import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

# To this:
import os
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    MODEL_DIR,
    PLOT_DIR,
    DEVICE,
    FEATURE_SIZE,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    LR,
    CLIP_GRAD,
    BETA,
)
from dataset import GrooveDataset
from models.vae import MusicVAE


def check_dataset(split_name):
    dataset = GrooveDataset(split=split_name)
    print(split_name, "size:", len(dataset))

    if len(dataset) == 0:
        raise ValueError(split_name + " dataset is empty.")

    x = dataset[0]
    print(split_name, "sample shape:", tuple(x.shape))
    print(split_name, "value range:", float(x.min()), "to", float(x.max()))
    return dataset


def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_items = 0

    for batch in loader:
        x = batch.to(DEVICE)

        if optimizer is not None:
            optimizer.zero_grad()

        recon, mu, logvar = model(x)
        recon_loss = criterion(recon, x)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + (BETA * kl_loss)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_items += batch_size

    avg_loss = total_loss / total_items
    avg_recon = total_recon / total_items
    avg_kl = total_kl / total_items
    return avg_loss, avg_recon, avg_kl


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    train_dataset = check_dataset("train")
    val_dataset = check_dataset("val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MusicVAE(
        input_dim=FEATURE_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_kl_losses = []
    val_kl_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_recon, train_kl = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_recon, val_kl = run_epoch(model, val_loader, criterion, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recon_losses.append(train_recon)
        val_recon_losses.append(val_recon)
        train_kl_losses.append(train_kl)
        val_kl_losses.append(val_kl)

        print(
            "Epoch {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f} | "
            "Train Recon: {:.6f} | Val Recon: {:.6f} | Train KL: {:.6f} | Val KL: {:.6f}".format(
                epoch, EPOCHS, train_loss, val_loss, train_recon, val_recon, train_kl, val_kl
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_DIR, "best_vae.pt")
            torch.save(model.state_dict(), save_path)
            print("Saved best model to:", save_path)

    # Total loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 2 VAE Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(PLOT_DIR, "vae_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Reconstruction loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_recon_losses, label="Train Recon")
    plt.plot(val_recon_losses, label="Val Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Task 2 VAE Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    recon_plot_path = os.path.join(PLOT_DIR, "vae_recon_loss_curve.png")
    plt.savefig(recon_plot_path)
    plt.close()

    # KL loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_kl_losses, label="Train KL")
    plt.plot(val_kl_losses, label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("Task 2 VAE KL Loss")
    plt.legend()
    plt.tight_layout()
    kl_plot_path = os.path.join(PLOT_DIR, "vae_kl_loss_curve.png")
    plt.savefig(kl_plot_path)
    plt.close()

    print("Training complete.")
    print("Best val loss:", best_val_loss)
    print("Saved plots:")
    print(loss_plot_path)
    print(recon_plot_path)
    print(kl_plot_path)


if __name__ == "__main__":
    main()
