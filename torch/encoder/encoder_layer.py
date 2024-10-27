"""Encoder layer module."""

from typing import Optional

from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    Implement the Encoder layer module.

    Args:
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        d_ff: The dimension of the feed-forward network.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implement the Forward pass of the encoder layer.

        Args:
            src: (seq_len, batch_size, d_model)
            src_mask: (batch_size, 1, seq_len, seq_len)

        Returns
        -------
            Tensor after encoder layer.
        """
        # Self-attention sublayer
        attn_output = self.self_attn(src, src, src, src_mask)  # (seq_len, batch_size, d_model)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feed-forward sublayer
        ff_output = self.feed_forward(src)  # (seq_len, batch_size, d_model)
        src = src + self.dropout(ff_output)
        return self.norm2(src)
