"""Transformer model."""

from typing import Optional

from decoder import Decoder
from encoder import Encoder

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Implement the Transformer model.

    Args:
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        d_ff: The dimension of the feed-forward network.
        input_vocab_size: Size of the input vocabulary.
        target_vocab_size: Size of the target vocabulary.
        max_seq_length: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        max_seq_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            input_vocab_size=input_vocab_size,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            target_vocab_size=target_vocab_size,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )
        self.out = nn.Linear(d_model, target_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implement the Forward pass of the transformer model.

        Args:
            src: (src_seq_len, batch_size)
            tgt: (tgt_seq_len, batch_size)
            src_mask: (batch_size, 1, src_seq_len, src_seq_len)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: (batch_size, 1, tgt_seq_len, src_seq_len)

        Returns
        -------
            Output logits.
        """
        memory = self.encoder(src, src_mask)  # (src_seq_len, batch_size, d_model)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)  # (tgt_seq_len, batch_size, d_model)
        return self.out(output)  # (tgt_seq_len, batch_size, target_vocab_size)
