import torch
import torch.nn as nn
from typing_extensions import Union


class SplitDepthSequenceNet(nn.Module):
    def __init__(self, in_channel: int, stride: int, out_channel: int, num_heads: int,
                 kernel: Union[int, tuple[int, int]], padding: Union[int, tuple[int, int]]):
        super(SplitDepthSequenceNet, self).__init__()
        if in_channel % num_heads != 0 or out_channel % num_heads != 0:
            exit(
                f"sub - convolutional layer in_channel(out_channel) is must be return integer (not float) "
                f"\n[ args[in_channel, out_channel, num_heads ]=[{in_channel}, {out_channel}, {num_heads}]"
                f" sub-conv in_channel={in_channel / num_heads} out_channel={out_channel / num_heads} ]")
        in_c = int(in_channel / num_heads)
        out_c = int(out_channel / num_heads)

        self.loop = num_heads

        self.conv2D = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, stride=stride,
                                padding=padding)

    def forward(self, x):

        stack = []
        for i in range(self.loop):
            _min = i * self.conv2D.in_channels
            _max = (i + 1) * self.conv2D.in_channels
            x1 = x[:, _min:_max, :, :]
            x1 = self.conv2D(x1)
            stack.append(x1)
        x = torch.cat(stack, dim=1)
        return x
