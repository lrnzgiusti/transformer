"""Feed-forward network in the transformer model."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Implement the Feed-forward network in the transformer model.

    Args:
        d_model: The dimension of the embeddings.
        d_ff: The dimension of the feed-forward network.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the Forward pass of the feed-forward network.

        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns
        -------
            Tensor after feed-forward network.
        """
        return self.linear2(self.dropout(nn.functional.relu(self.linear1(x))))
