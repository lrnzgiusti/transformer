"""Positional encoding module for Transformer."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implement the Positional encoding module for Transformer.

    Args:
        d_model: The dimension of the embeddings.
        max_len: The maximum length of input sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create a long enough P
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the Forward pass of the positional encoding module.

        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns
        -------
            Tensor with positional encoding added.
        """
        return x + self.pe[: x.size(0), :]
