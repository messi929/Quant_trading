"""Temporal Transformer for time-series pattern discovery with sector cross-attention.

Input: VAE latent indicator sequences
Output: Time-aware signal embeddings and future return predictions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal ordering."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SectorCrossAttention(nn.Module):
    """Cross-attention between sector embeddings and sequence features.

    Captures inter-sector dynamics and market-wide signals.
    """

    def __init__(self, d_model: int, n_sectors: int, n_heads: int = 4):
        super().__init__()
        self.sector_embeddings = nn.Embedding(n_sectors, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        sector_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) sequence features
            sector_ids: (batch,) integer sector IDs

        Returns:
            (batch, seq_len, d_model) sector-enriched features
        """
        # Get sector embeddings for the batch
        sector_emb = self.sector_embeddings(sector_ids)  # (batch, d_model)
        sector_emb = sector_emb.unsqueeze(1)  # (batch, 1, d_model)

        # Also include all sector embeddings as context
        all_sectors = self.sector_embeddings.weight.unsqueeze(0).expand(
            x.size(0), -1, -1
        )  # (batch, n_sectors, d_model)

        # Cross-attend: sequence queries, sector keys/values
        attn_out, _ = self.cross_attn(x, all_sectors, all_sectors)
        return self.norm(x + attn_out)


class TemporalTransformer(nn.Module):
    """Transformer for discovering temporal patterns in latent indicators.

    Architecture:
        Input projection → Positional encoding → Transformer encoder
        → Sector cross-attention → Output head (return prediction)

    The attention mechanism reveals which time steps and which
    latent indicators are most important for predictions.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 60,
        output_dim: int = 1,
        n_sectors: int = 11,
        use_sector_attention: bool = True,
        sector_embed_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_sector_attention = use_sector_attention

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Sector cross-attention
        if use_sector_attention:
            self.sector_attention = SectorCrossAttention(
                d_model, n_sectors, n_heads=4
            )

        # Sector embedding for concatenation
        self.sector_embed = nn.Embedding(n_sectors, sector_embed_dim)

        # Output head
        head_input_dim = d_model + sector_embed_dim
        self.output_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        # Signal embedding head (for downstream use by RL agent)
        self.signal_head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        sector_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input sequences
            sector_ids: (batch,) sector integer IDs
            return_attention: Whether to return attention weights

        Returns:
            Dict with keys: prediction, signal_embedding, [attention_weights]
        """
        # Input projection + positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Transformer encoding
        h = self.transformer(h)

        # Sector cross-attention
        if self.use_sector_attention:
            h = self.sector_attention(h, sector_ids)

        # Pool: use last time step (causal)
        h_last = h[:, -1, :]  # (batch, d_model)

        # Concatenate sector embedding
        sector_emb = self.sector_embed(sector_ids)  # (batch, sector_embed_dim)
        h_combined = torch.cat([h_last, sector_emb], dim=-1)

        # Output heads
        prediction = self.output_head(h_combined)
        signal_embedding = self.signal_head(h_combined)

        result = {
            "prediction": prediction.squeeze(-1),
            "signal_embedding": signal_embedding,
        }

        if return_attention:
            # Extract attention weights from last layer
            result["temporal_features"] = h

        return result

    def get_signal_embedding(
        self,
        x: torch.Tensor,
        sector_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get signal embedding for RL agent consumption."""
        with torch.no_grad():
            output = self.forward(x, sector_ids)
        return output["signal_embedding"]
