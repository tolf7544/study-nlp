import torch
from torch import nn


# https://m.blog.naver.com/heianjung/222619364695 # temperature 개념
# https://www.linkedin.com/pulse/overcoming-limitations-softmax-sharp-performance-ai-systems-wendin-ubupf # softmax와 temperature에 대한 정보 부분 참조
# https://discuss.pytorch.org/t/backward-temperature-softmax-implementation/62133  # temperature 적용 코드 참조

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int, dropout_scale):
        super(AttentionPooling, self).__init__()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_scale)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32)).data.clamp(0.01, 2.0)

    def forward(self, x):
        score = x / self.temperature
        score = self.softmax(score)
        score = self.dropout(score)
        x1 = torch.sum(x * score, dim=-1)
        x1 = x1.unsqueeze(-1)
        x = x1 + x
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x
