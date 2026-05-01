# src/training/train_vae_v2.py
import os, sys, torch
import torch.nn as nn
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
from models.vae import MusicVAE


# ── KL Annealing — gradually increase BETA over epochs ────────
# This prevents KL collapse by starting with low KL weight
# and slowly increasing it so model learns reconstruction first
def get_beta(epoch, total_epochs, beta_max=4.0):
    """Linear KL annealing from 0 to beta_max."""
    return beta_max * (epoch / total_epochs)


def weighted_recon_loss(output, target, pos_weight=20.0):
    """Penalize missed notes more than silence."""
    weights = torch.where(
        target > 0.15,
        torch.tensor(pos_weight, device=target.device),
        torch.tensor(1.0,        device=target.device)
    )
    return (weights * (output - target) ** 2).mean()


def kl_divergence(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def run_epoch(model, loader, optimizer=None, beta=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_recon = total_kl = total_items = 0

    for batch in loader:
        x = batch.to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        recon, mu, logvar = model(x)
        recon_loss = weighted_recon_loss(recon, x)
        kl_loss    = kl_divergence(mu, logvar)
        loss       = recon_loss + beta * kl_loss

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        bs           = x.size(0)
        total_loss  += loss.item()       * bs
        total_recon += recon_loss.item() * bs
        total_kl    += kl_loss.item()    * bs
        total_items += bs

    n = total_items
    return total_loss/n, total_recon/n, total_kl/n


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    train_dataset = GrooveDataset(split='train')
    val_dataset   = GrooveDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train samples: {len(train_dataset):,}')
    print(f'Val   samples: {len(val_dataset):,}')

    model = MusicVAE(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    all_train_loss  = []
    all_val_loss    = []
    all_train_recon = []
    all_val_recon   = []
    all_train_kl    = []
    all_val_kl      = []
    all_betas       = []

    best_path = os.path.join(MODEL_DIR, 'best_vae_v2.pt')

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch, EPOCHS, beta_max=4.0)
        all_betas.append(beta)

        train_loss, train_recon, train_kl = run_epoch(model, train_loader, optimizer, beta)
        val_loss,   val_recon,   val_kl   = run_epoch(model, val_loader,   None,      beta)

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_recon.append(train_recon)
        all_val_recon.append(val_recon)
        all_train_kl.append(train_kl)
        all_val_kl.append(val_kl)

        print(
            f'Epoch {epoch:02d}/{EPOCHS} | β={beta:.2f} | '
            f'Loss={train_loss:.5f}/{val_loss:.5f} | '
            f'Recon={train_recon:.5f}/{val_recon:.5f} | '
            f'KL={train_kl:.5f}/{val_kl:.5f}'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f'  ✅ Saved best model → {best_path}')

    # Plot all curves
    epochs_range = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 2 VAE v2 — Training Curves (KL Annealing + Weighted MSE)', fontsize=13)

    axes[0,0].plot(epochs_range, all_train_loss,  label='Train')
    axes[0,0].plot(epochs_range, all_val_loss,    label='Val')
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(epochs_range, all_train_recon, label='Train')
    axes[0,1].plot(epochs_range, all_val_recon,   label='Val')
    axes[0,1].set_title('Reconstruction Loss (Weighted MSE)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(epochs_range, all_train_kl,    label='Train')
    axes[1,0].plot(epochs_range, all_val_kl,      label='Val')
    axes[1,0].set_title('KL Divergence Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(epochs_range, all_betas, color='green')
    axes[1,1].set_title('KL Annealing Schedule (β)')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('β value')
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'vae_v2_loss_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'\n📊 Plots saved → {plot_path}')
    print(f'🏆 Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()