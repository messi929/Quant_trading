"""VolTransformer: AlphaTransformer adapted for volatility expansion prediction.

Architecture:
    Input projection → Positional encoding → Transformer encoder
    → Mean pooling → Prediction head (vol expansion) + Confidence head
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class VolTransformer(nn.Module):
    """Transformer for volatility expansion prediction.

    Predicts: "Will this ticker's volatility expand in the next N days?"
    - prediction: scalar vol expansion ratio (>0 = expanding)
    - confidence: P(prediction is reliable) in [0, 1]
    - embedding: d_model vector for downstream use
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 5,
        d_ff: int = 768,
        dropout: float = 0.1,
        max_seq_length: int = 60,
        use_confidence_head: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_confidence_head = use_confidence_head

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Prediction head: vol expansion scalar
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Confidence head: P(prediction reliable)
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
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) — feature sequences

        Returns:
            {"prediction": (batch,), "confidence": (batch,), "embedding": (batch, d_model)}
        """
        # Input projection + positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Transformer
        h = self.transformer(h)

        # Mean pooling over sequence (V2 lesson: better than last-token)
        h_pooled = h.mean(dim=1)  # (batch, d_model)

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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
