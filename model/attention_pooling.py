import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super(AttentionPooling, self).__init__()
        self.gelu = nn.GELU
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    # https://www.linkedin.com/pulse/overcoming-limitations-softmax-sharp-performance-ai-systems-wendin-ubupf 부분 참조
    def forward(self, x):
        score = self.softmax(
            x / self.temperature)  # https://discuss.pytorch.org/t/backward-temperature-softmax-implementation/62133
        x1 = torch.sum(x * score, dim=2)
        x1 = x1.unsqueeze(-1)
        x = x1 + x
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x