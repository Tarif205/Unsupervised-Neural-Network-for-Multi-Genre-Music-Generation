# src/training/train_vae.py
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
from models.vae import MusicVAE


def focal_loss(logits, targets, gamma=2.0, pos_weight=84.0):
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction='none'
    )
    probs   = torch.sigmoid(logits)
    pt      = torch.where(targets > 0.15, probs, 1 - probs)
    focal_w = (1 - pt) ** gamma
    return (focal_w * bce_loss).mean()


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def get_beta(epoch, warmup=10, beta_max=1.0, total=30):
    """KL annealing: 0 during warmup, linear increase after."""
    if epoch <= warmup:
        return 0.0
    return beta_max * (epoch - warmup) / (total - warmup)


def run_epoch(model, loader, optimizer=None, beta=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_recon = total_kl = total_items = 0

    for batch in loader:
        x = batch.to(DEVICE)
        if is_train:
            optimizer.zero_grad()

        logits, mu, logvar = model(x)
        recon = focal_loss(logits, x)
        kl    = kl_divergence(mu, logvar)
        loss  = recon + beta * kl

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        bs           = x.size(0)
        total_loss  += loss.item()  * bs
        total_recon += recon.item() * bs
        total_kl    += kl.item()    * bs
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

    print(f'Train: {len(train_dataset):,} | Val: {len(val_dataset):,}')

    model = MusicVAE(
        input_dim  = FEATURE_SIZE,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    best_val   = float('inf')

    # ── same name as before ────────────────────────────────────
    best_path  = os.path.join(MODEL_DIR, 'best_vae.pt')

    all_loss  = []
    all_recon = []
    all_kl    = []
    all_beta  = []

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch, warmup=10, beta_max=1.0, total=EPOCHS)
        all_beta.append(beta)

        tr_loss, tr_recon, tr_kl = run_epoch(model, train_loader, optimizer, beta)
        vl_loss, vl_recon, vl_kl = run_epoch(model, val_loader,   None,      beta)

        all_loss.append((tr_loss, vl_loss))
        all_recon.append((tr_recon, vl_recon))
        all_kl.append((tr_kl, vl_kl))

        print(
            f'Epoch {epoch:02d}/{EPOCHS} | β={beta:.2f} | '
            f'Loss={tr_loss:.5f}/{vl_loss:.5f} | '
            f'Recon={tr_recon:.5f}/{vl_recon:.5f} | '
            f'KL={tr_kl:.5f}/{vl_kl:.5f}'
        )

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), best_path)
            print(f'  Saved best model → {best_path}')

    # ── Plots ──────────────────────────────────────────────────
    epochs_r = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Task 2 VAE — Focal Loss + KL Annealing', fontsize=13)

    axes[0].plot(epochs_r, [x[0] for x in all_loss],  label='Train', color='steelblue')
    axes[0].plot(epochs_r, [x[1] for x in all_loss],  label='Val',   color='coral', linestyle='--')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_r, [x[0] for x in all_kl], label='Train KL', color='steelblue')
    axes[1].plot(epochs_r, [x[1] for x in all_kl], label='Val KL',   color='coral', linestyle='--')
    axes[1].set_title('KL Divergence')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_r, all_beta, color='green', linewidth=2)
    axes[2].set_title('KL Annealing Schedule (β)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('β')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'task2_vae_loss.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'\nTraining complete!')
    print(f'   Best val loss : {best_val:.6f}')
    print(f'   Plot saved    : {plot_path}')


if __name__ == '__main__':
    main()