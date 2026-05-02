# src/models/autoencoder_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification on sparse piano rolls.
    Suppresses gradient from easy (silent) cells.
    """
    def __init__(self, gamma=2.0, pos_weight=20.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: raw output (no sigmoid applied)
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device)
        )
        probs      = torch.sigmoid(logits)
        pt         = torch.where(targets == 1, probs, 1 - probs)
        focal_w    = (1 - pt) ** self.gamma
        focal_loss = focal_w * F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        return focal_loss.mean()


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        enc_drop = dropout if num_layers > 1 else 0.0
        dec_drop = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = enc_drop,
        )

        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # ── Decoder input = hidden + z concatenated ──────────────
        # Faculty fix: concatenate z at every decoder step
        self.decoder = nn.LSTM(
            input_size  = hidden_dim + latent_dim,  # ← concatenate z
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dec_drop,
        )

        # ── NO sigmoid here — raw logits for BCEWithLogitsLoss ───
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        last_hidden    = hidden[-1]
        latent         = self.hidden_to_latent(last_hidden)
        return latent

    def decode(self, z, seq_len):
        hidden   = self.latent_to_hidden(z)
        repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # ── Concatenate z at every time step ─────────────────────
        z_repeated     = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input  = torch.cat([repeated, z_repeated], dim=-1)

        decoded, _     = self.decoder(decoder_input)
        logits         = self.output_layer(decoded)
        return logits   # raw logits — NO sigmoid

    def forward(self, x):
        seq_len = x.size(1)
        latent  = self.encode(x)
        logits  = self.decode(latent, seq_len)
        return logits