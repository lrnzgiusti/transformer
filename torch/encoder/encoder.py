"""Encoder module."""

import math
from typing import Optional

from pos_enc import PositionalEncoding

import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Implement the Encoder layer of the transformer model.

    Args:
        num_layers: Number of encoder layers.
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        d_ff: The dimension of the feed-forward network.
        input_vocab_size: Size of the input vocabulary.
        max_seq_length: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implement the Forward pass of the encoder module.

        Args:
            src: (seq_len, batch_size)
            src_mask: (batch_size, 1, seq_len, seq_len)

        Returns
        -------
            Encoder outputs.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)  # (seq_len, batch_size, d_model)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
