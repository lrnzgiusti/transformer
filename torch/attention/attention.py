"""Attention module for the Transformer model."""

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Implement the Multi-head attention module.

    Args:
        d_model: The dimension of the embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads == 0:
            msg = "d_model must be divisible by num_heads."
            raise ValueError(msg)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define linear layers for query, key, and value
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implement the Forward pass of the multi-head attention layer.

        Args:
            query: (q_seq_len, batch_size, d_model)
            key: (k_seq_len, batch_size, d_model)
            value: (v_seq_len, batch_size, d_model)
            mask: (batch_size, 1, q_seq_len, k_seq_len)

        Returns
        -------
            Output tensor after attention.
        """
        q_seq_len, batch_size, _ = query.size()
        k_seq_len, _, _ = key.size()
        v_seq_len, _, _ = value.size()

        # Linear projections
        q = (
            self.linear_q(query).view(q_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        )  # (batch_size, num_heads, q_seq_len, d_k)
        k = (
            self.linear_k(key).view(k_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        )  # (batch_size, num_heads, k_seq_len, d_k)
        v = (
            self.linear_v(value).view(v_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        )  # (batch_size, num_heads, v_seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (batch_size, num_heads, q_seq_len, k_seq_len)

        if mask is not None:
            # Ensure mask is broadcastable to (batch_size, num_heads, q_seq_len, k_seq_len)
            scores = scores.masked_fill(mask == float("-inf"), -1e9)

        attn = nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute attention output
        context = torch.matmul(attn, v)  # (batch_size, num_heads, q_seq_len, d_k)
        context = (
            context.permute(2, 0, 1, 3).contiguous().view(q_seq_len, batch_size, self.d_model)
        )  # (q_seq_len, batch_size, d_model)

        # Final linear layer
        return self.linear_out(context)  # (q_seq_len, batch_size, d_model)
