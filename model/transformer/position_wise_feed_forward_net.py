import torch
import torch.nn as nn


class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_model: int, d_pwff: int, dropout_scale: int):
        super(PositionWiseFeedForwardNetworks, self).__init__()

        self.fc1 = nn.Linear(d_model, d_pwff)
        self.fc2 = nn.Linear(d_pwff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_scale)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
