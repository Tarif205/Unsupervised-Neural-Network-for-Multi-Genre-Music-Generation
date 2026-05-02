# src/models/vae_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicVAE(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.latent_dim = latent_dim
        enc_drop = dropout if num_layers > 1 else 0.0
        dec_drop = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = enc_drop,
        )

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # ── Concatenate z at every decoder step ──────────────────
        self.decoder = nn.LSTM(
            input_size  = hidden_dim + latent_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dec_drop,
        )

        # ── NO sigmoid — raw logits ───────────────────────────────
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        last_hidden    = hidden[-1]
        mu             = self.fc_mu(last_hidden)
        logvar         = self.fc_logvar(last_hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        hidden     = self.latent_to_hidden(z)
        repeated   = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate z at every time step
        decoder_input = torch.cat([repeated, z_repeated], dim=-1)
        decoded, _    = self.decoder(decoder_input)
        logits        = self.output_layer(decoded)
        return logits  # raw logits

    def forward(self, x):
        seq_len        = x.size(1)
        mu, logvar     = self.encode(x)
        z              = self.reparameterize(mu, logvar)
        logits         = self.decode(z, seq_len)
        return logits, mu, logvar

    def sample(self, num_samples, seq_len, device):
        z      = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decode(z, seq_len)
        return torch.sigmoid(logits)  # sigmoid only at inference