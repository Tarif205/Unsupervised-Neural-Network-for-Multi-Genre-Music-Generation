
import torch
import torch.nn as nn


class MusicVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        latent_dim=128,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        enc_dropout = dropout if num_layers > 1 else 0.0
        dec_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Sigmoid()

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        last_hidden = hidden[-1]
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, seq_len):
        hidden_base = self.latent_to_hidden(z)
        decoder_input = hidden_base.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        out = self.output_layer(decoded)
        out = self.output_activation(out)
        return out

    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar

    def sample(self, num_samples, seq_len, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, seq_len)
        return samples
