import math

import torch
import torch.nn as nn
# https://www.geeksforgeeks.org/nlp/positional-encoding-in-transformers/
# https://cpm0722.github.io/pytorch-implementation/transformer
# https://tigris-data-science.tistory.com/entry/%EC%B0%A8%EA%B7%BC%EC%B0%A8%EA%B7%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-Transformer5-Positional-Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, sequence_length, device):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
        i = torch.arange(0, d_model, dtype=torch.float32, device=device).unsqueeze(0)
        angle_rads = position / torch.float_power(10000, 2*(i // 2) / d_model)
        angle_rads.to(device)
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        self.encoding = angle_rads.unsqueeze(0)
        self.encoding.requires_grad = False

    def forward(self, x):
        return x + self.encoding.type(torch.float32)

# if __name__ == '__main__':
# # case 1
#     sequence_length = 10
#     d_model = 2
#     position = torch.arange(0, sequence_length, dtype=torch.float32).unsqueeze(1)
#     i = torch.arange(0, d_model, dtype=torch.float32).unsqueeze(0)
#     angle_rads = position / torch.float_power(10000, 2*(i // 2) / d_model)
#
#     angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
#     angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
#
#     encoding_1 = angle_rads.unsqueeze(0)
#
#     print(encoding_1.shape)

#
# case 2
#     encoding = torch.zeros(sequence_length, d_model)
#     encoding.requires_grad = False
#     position = torch.arange(0, sequence_length).float().unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#     encoding[:, 0::2] = torch.sin(position * div_term)
#     encoding[:, 1::2] = torch.cos(position * div_term)
#     encoding_2 = encoding.unsqueeze(0)
#
#     print(torch.allclose(encoding_1.float(), encoding_2.float()))