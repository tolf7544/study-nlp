import torch.nn as nn

from model.transformer.positional_encoding import PositionalEncoding
from model.transformer.transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_pwff, sequence_length, device, num_layers, num_heads,
                 transformer_dropout_scale: int = 0.2, pwff_dropout_scale: int = 0.2):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, sequence_length=sequence_length, device=device)
        self.num_layers = num_layers
        self.transformer_encoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_encoder_blocks.append(
                TransformerEncoderBlock(d_model, d_pwff, num_heads, transformer_dropout_scale, pwff_dropout_scale, device=device)
            )

    def forward(self, x, pad_mask):
        x = self.positional_encoding(x)
        for encoder_block in self.transformer_encoder_blocks:
            x = encoder_block(x, pad_mask)
        return x
