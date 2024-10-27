"""Decoder layer module."""

from typing import Optional

from attention import MultiHeadAttention
from feed_forward import PositionwiseFeedForward

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    """
    Implement the Decoder layer module.

    Args:
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        d_ff: The dimension of the feed-forward network.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implement the Forward pass of the decoder layer.

        Args:
            tgt: (tgt_seq_len, batch_size, d_model)
            memory: (src_seq_len, batch_size, d_model)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: (batch_size, 1, tgt_seq_len, src_seq_len)

        Returns
        -------
            Tensor after decoder layer.
        """
        # Self-attention sublayer
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)  # (tgt_seq_len, batch_size, d_model)
        tgt = tgt + self.dropout(self_attn_output)
        tgt = self.norm1(tgt)

        # Cross-attention sublayer
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)  # (tgt_seq_len, batch_size, d_model)
        tgt = tgt + self.dropout(cross_attn_output)
        tgt = self.norm2(tgt)

        # Feed-forward sublayer
        ff_output = self.feed_forward(tgt)  # (tgt_seq_len, batch_size, d_model)
        tgt = tgt + self.dropout(ff_output)
        return self.norm3(tgt)
