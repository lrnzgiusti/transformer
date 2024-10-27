"""Decoder module for the Transformer model."""

import math
from typing import Optional

from pos_enc import PositionalEncoding

import torch
import torch.nn as nn

from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Implement the Decoder module for the Transformer model.

    Args:
        num_layers: Number of decoder layers.
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        d_ff: The dimension of the feed-forward network.
        target_vocab_size: Size of the target vocabulary.
        max_seq_length: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        target_vocab_size: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implement the Forward pass of the decoder module.

        Args:
            tgt: (tgt_seq_len, batch_size)
            memory: (src_seq_len, batch_size, d_model)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: (batch_size, 1, tgt_seq_len, src_seq_len)

        Returns
        -------
            Decoder outputs.
        """
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # (tgt_seq_len, batch_size, d_model)
        tgt = self.pos_encoder(tgt)  # (tgt_seq_len, batch_size, d_model)
        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        return tgt
