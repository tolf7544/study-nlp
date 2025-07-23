import torch
import torch.nn as nn
from torch import Tensor

from model.jamo_cnn.split_depth_sequence_cnn import SplitDepthSequenceNet

# #
# mask 과정 예시
# #
# a = torch.ones((10, 10))  # mask 적용 대상
# mask = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 1, 0]) # length 10인 mask
# mask = mask.unsqueeze(0) # batch_size를 고려한 shape 매칭
# mask = mask * (1 / 3) + (1 - mask) ( 1 - mask 과정은 broadcasting을 통해 [1-1, 1-1, 1-1, 1-0....] 즉 [0, 0, 0, 1....] 로 되기에 NOT 논리 연산처럼 작용한다.
#   mask * (1/3)은 기존의 다운 스케일 부분의 계수를 적용한다/
#   즉 둘이 합쳐 자모는 1로 유지되며, 자모 외 영역은 다운 스케일이 적용된다.

# 아래는 두개의 broadcasting을 아용하여 직관적으로 수행한 예시이다.
# print(a * mask)
# shape = mask.shape[0]
# x1 = torch.ones((shape,shape))
# x2 = torch.ones((shape,shape))
# x1 = x1*mask*(1/3)
# x2 = x2*(mask*-1+1)
# print(x2+x1)

class JamoCNN(nn.Module):
    def __init__(self, sequence_length: int, num_heads: int, dropout: int, device):
        super(JamoCNN, self).__init__()
        self.fc = nn.Linear(256, 512)
        self.gelu = nn.GELU()
        self.batch_norm1d = nn.BatchNorm1d(sequence_length)  # sds_conv 와 이어져 진행됨

        self.sds_conv = SplitDepthSequenceNet(in_channel=sequence_length, out_channel=sequence_length,
                                              num_heads=num_heads,
                                              kernel=(5, 3), stride=(1, 3),
                                              padding=(2, 0))  # 후의 transformer와 연동하기 위해 embedding_dim을 out_channel로 지정

        # 해당 파라미터(계수)는 mask의 스케일(적용 강도)로 입력값에 따른 디테일한 대응을 할 수 있도록 learnable 하게 선언함.
        self.down_scale = nn.Parameter(torch.tensor(1 / 3, dtype=torch.float32, device=device)).clamp(0.01, 1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, down_scaling_mask):
        x: Tensor = x.transpose(-2, -1)
        x: Tensor = self.sds_conv(x)
        x: Tensor = x.transpose(-2, -1)
        x = x.squeeze(-2)

        down_scaling_mask = down_scaling_mask.unsqueeze(-1)
        down_scaling_mask = down_scaling_mask * self.down_scale + (1.0 - down_scaling_mask)
        x = x * down_scaling_mask

        x = self.batch_norm1d(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
