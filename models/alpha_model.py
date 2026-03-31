"""AlphaTransformer — Conviction-based return prediction model (v2).

Key changes from v1 TemporalTransformer:
  1. Mean pooling (uses full sequence, not just last token)
  2. Confidence head (predicts whether direction prediction is correct)
  3. No sector cross-attention (predicts across all stocks, not within sectors)
  4. Simpler architecture (single model, no ensemble)

Input:  (batch, seq_len, n_features) z-score normalized features
Output: {prediction: (batch,), confidence: (batch,), embedding: (batch, d_model)}
"""

import math

import torch
import torch.nn as nn


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


class AlphaTransformer(nn.Module):
    """Transformer for per-ticker return prediction with confidence.

    Architecture:
        Input projection → Positional encoding → Transformer encoder
        → Mean pooling → Prediction head + Confidence head
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
        use_confidence_head: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_confidence_head = use_confidence_head

        # Input projection: raw features → d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Final layer norm after pooling
        self.pool_norm = nn.LayerNorm(d_model)

        # Prediction head: scalar return prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Confidence head: P(direction prediction is correct)
        if use_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # Confidence head: initialize bias so initial confidence ≈ 0.5
            if "confidence_head" in name and "bias" in name and p.dim() == 1:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) z-score normalized features

        Returns:
            Dict with keys:
                prediction: (batch,) return prediction
                confidence: (batch,) P(direction correct), only if use_confidence_head
                embedding:  (batch, d_model) pooled representation
        """
        # Input projection + positional encoding
        h = self.input_proj(x)      # (batch, seq_len, d_model)
        h = self.pos_encoding(h)

        # Transformer encoding
        h = self.transformer(h)     # (batch, seq_len, d_model)

        # Mean pooling over sequence (v2: uses ALL timesteps)
        h_pooled = h.mean(dim=1)    # (batch, d_model)
        h_pooled = self.pool_norm(h_pooled)

        # Prediction
        prediction = self.prediction_head(h_pooled).squeeze(-1)  # (batch,)

        result = {
            "prediction": prediction,
            "embedding": h_pooled,
        }

        # Confidence
        if self.use_confidence_head:
            confidence = self.confidence_head(h_pooled).squeeze(-1)  # (batch,)
            result["confidence"] = confidence

        return result
