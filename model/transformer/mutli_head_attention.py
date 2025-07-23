import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads

        self.d_model = d_model
        self.d_qkv = d_model // num_heads

        self.d_qkv_scale = torch.tensor(math.sqrt(self.d_qkv), dtype=torch.float32, device=device)

        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

        _scale = torch.tensor([1.0 for _ in range(num_heads)], dtype=torch.float32, device=device)
        self.temperature = nn.Parameter(_scale).clamp(0.01, 2.0)

    def transform_mini_batch(self, x: Tensor):
        batch_size, sequence_length, d_model = x.shape
        # transpose는 contiguous이지 않는다.
        # view는 연속적인 메모리에만 정산적인 작업 수행이 가능하다.
        x = x.view(batch_size, sequence_length, self.num_heads, self.d_qkv)
        x = x.transpose(1, 2)
        return x

    def scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]):
        score = torch.matmul(query, key.transpose(dim0=-2, dim1=-1))

        if mask is not None:
            mask = mask.unsqueeze(-2).unsqueeze(-2)
            score = score.masked_fill(mask == 0, -1e9)

        score = score / self.d_qkv_scale * self.temperature.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        score = self.softmax(score)
        score = self.dropout(score)
        return torch.matmul(score, value)


    def forward(self, x, mask):
        query = self.transform_mini_batch(self.fc_query(x))
        key = self.transform_mini_batch(self.fc_key(x))
        value = self.transform_mini_batch(self.fc_value(x))

        x = self.scaled_dot_product_attention(query, key, value, mask)
        x = x.transpose(1,2).contiguous()

        x = x.view(x.shape[0], x.shape[1], self.d_model)

        x = self.fc_out(x)
        return x
