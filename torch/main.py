"""Example of using the Transformer model in PyTorch."""

from transformer import Transformer

import torch


def generate_square_subsequent_mask(sz1: int, sz2: int) -> torch.Tensor:
    """Generate a mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = (torch.triu(torch.ones(sz1, sz2)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


# Hyperparameters
num_encoder_layers = 6
num_decoder_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
input_vocab_size = 10000
target_vocab_size = 10000
max_seq_length = 100
dropout = 0.1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = Transformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_seq_length=max_seq_length,
    dropout=dropout,
).to(device)

# Example input (source and target sequences)
# Shape: (seq_len, batch_size)
src = torch.randint(0, input_vocab_size, (35, 32)).to(device)  # (35, 32)
tgt = torch.randint(0, target_vocab_size, (20, 32)).to(device)  # (20, 32)

# Create masks
src_mask = None  # Modify if needed
# For target mask (self-attention in decoder)
tgt_seq_len = tgt.size(0)
tgt_mask = generate_square_subsequent_mask(tgt_seq_len, tgt_seq_len).to(device)

# For memory mask (cross-attention in decoder)
src_seq_len = src.size(0)
memory_mask = None  # Modify if needed


# Forward pass
output = model(src, tgt, src_mask, tgt_mask)

print(f"Output shape: {output.shape}")  # Expected: (20, 32, 10000)
