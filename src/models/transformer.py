# src/models/transformer.py
"""
Task 3 — GPT-style Transformer Decoder for Music Generation

Faculty guide:
- Decoder-only Transformer (GPT-style)
- Token embedding + positional embedding
- 4-8 Transformer decoder layers
- Masked multi-head self-attention (causal)
- Final linear projection to vocab size
- Autoregressive: p(x_t | x_{<t})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    Faculty: "causal mask prevents attending to future tokens"
    """
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=512):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Causal mask — upper triangular = future tokens
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        # Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                   for t in qkv]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_len=512):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """
    GPT-style decoder-only Transformer for music generation.

    Faculty guide:
    - Token embedding layer (vocab → d_model)
    - Positional embedding layer
    - Stack of N decoder layers with causal self-attention
    - Final linear projection to vocab size (logits)
    - NO softmax at output — use cross-entropy loss directly

    Input shape : (batch, seq_len)   integer token IDs
    Output shape: (batch, seq_len, vocab_size)  logits
    """

    def __init__(
        self,
        vocab_size  : int,
        d_model     : int   = 256,
        n_heads     : int   = 4,
        n_layers    : int   = 4,
        d_ff        : int   = 512,
        max_len     : int   = 512,
        dropout     : float = 0.1,
        pad_token   : int   = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.max_len    = max_len
        self.pad_token  = pad_token

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_len)
            for _ in range(n_layers)
        ])

        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())
        print(f"MusicTransformer | vocab={vocab_size} | d_model={d_model} "
              f"| layers={n_layers} | params={total:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        Args:
            idx: (B, T) integer token IDs

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        # Token + positional embeddings
        pos    = torch.arange(T, device=idx.device).unsqueeze(0)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        seed_tokens : torch.Tensor,
        max_new_tokens: int = 512,
        temperature : float = 1.0,
        top_k       : int   = 50,
    ):
        """
        Autoregressive generation with temperature + top-k sampling.

        Faculty guide:
        - "Start with seed token (e.g. Bar)"
        - "Sample next token from predicted distribution"
        - "Use temperature sampling to prevent repetition loops"
        - top-k: mask tokens outside top-k before sampling

        Args:
            seed_tokens : (1, T) starting token IDs
            max_new_tokens: number of new tokens to generate
            temperature : >1 = more random, <1 = more focused
            top_k       : only sample from top-k tokens

        Returns:
            (1, T + max_new_tokens) token IDs
        """
        self.eval()
        tokens = seed_tokens.clone()

        for _ in range(max_new_tokens):
            # Crop to max_len context
            ctx    = tokens[:, -self.max_len:]
            logits = self(ctx)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            tokens    = torch.cat([tokens, next_tok], dim=1)

        return tokens