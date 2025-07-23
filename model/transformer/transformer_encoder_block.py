import torch
import torch.nn as nn

from model.transformer.mutli_head_attention import MultiHeadSelfAttention
from model.transformer.position_wise_feed_forward_net import PositionWiseFeedForwardNetworks


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_pwff: int, num_heads: int, transformer_dropout_scale: int,
                 pwff_dropout_scale: int, device):
        super(TransformerEncoderBlock, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads, device=device)
        self.position_wise_feed_forward_networks = PositionWiseFeedForwardNetworks(d_model, d_pwff, pwff_dropout_scale)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(transformer_dropout_scale)

    def residual_connect(self, x, sub_layer):
        x1 = self.layer_norm(sub_layer)
        x1 = self.dropout(x1)
        x = x + x1
        return x

    def forward(self, x, mask):
        x = self.residual_connect(x, self.multi_head_self_attention(x, mask))
        x = self.residual_connect(x, self.position_wise_feed_forward_networks(x))
        return x
