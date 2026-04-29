#src/models/autoencoder.py
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Simple LSTM Autoencoder for piano-roll music windows.

    Input shape : (batch, seq_len, input_dim)
    Output shape: (batch, seq_len, input_dim)
    """

    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        encoder_dropout = dropout if num_layers > 1 else 0.0
        decoder_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=encoder_dropout,
        )

        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=decoder_dropout,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.output_activation = nn.Sigmoid()

    def encode(self, x):
        _, (hidden, _) = self.encoder(x)
        last_hidden = hidden[-1]
        latent = self.hidden_to_latent(last_hidden)
        return latent

    def decode(self, latent, seq_len):
        hidden = self.latent_to_hidden(latent)
        repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(repeated)
        output = self.output_layer(decoded)
        output = self.output_activation(output)
        return output

    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed
