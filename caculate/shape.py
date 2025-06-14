import numbers
from numbers import Number
from typing import Literal, Callable
import torch
from sympy import Integer
from torchgen.executorch.api.et_cpp import return_names


def print_layer(line: Literal["layer", "method"] = "layer", layer_name: str = None, shape: list[int] = None,
                shape_type: Literal["chunk", "layer"] = "layer", status: Literal["active", "end"] = "end"):
    print(f"{line}: {layer_name}, {shape_type} shape: {shape} status: {status}")


def print_model(model_name):
    print(f"model: {model_name}")


def reshape(shape: list[int] = None, index: int = None, target: int = None):
    temp = shape[target]
    shape[target] = shape[index]
    shape[index] = shape[temp]
    return shape

def unsqueeze(shape: list[int], dim: int = -1):
    if dim ** 2 / dim > len(shape):
        exit(
            f"unsqueeze dimension index is over than input shape dim. [ unsqueeze of args: {dim}, shape of args: {shape} ]")

    if dim < 0:
        shape.insert(len(shape) + dim + 1, 1)
    else:
        shape.insert(dim-1, 1)
    print_layer(layer_name="unsqueeze", line="method", shape=shape)
    return shape

def squeeze(shape: list[int], dim: int = -1):
    if dim ** 2 / dim > len(shape):
        exit(
            f"squeeze dimension index is over than input shape dim. [ squeeze of args: {dim}, shape of args: {shape} ]")
    shape.pop(dim)
    print_layer(layer_name="squeeze", shape=shape)
    return shape

def torch_sum(shape: list[int], dim):
    shape.pop(dim)
    print_layer(layer_name=f"torch_sum dim={dim}", line="method", shape=shape)
    return shape

def input_layer(shape: list[int]):
    print_layer(layer_name="input", shape=shape)
    return shape


def embedding(shape: list[int], d_model: int):
    shape.append(d_model)
    print_layer(layer_name="embedding", shape=shape)
    return shape


def fc(shape, out_shape):
    shape[-1] = out_shape
    print_layer(layer_name="fc", shape=shape)
    return shape


def conv2d(shape: list[int], in_channel: int = None, out_channel: int = None,
           kernel: tuple[int, int] = None,
           padding: tuple[int, int] = (0, 2), stride: tuple[int, int] = (1, 1),
           dilation: tuple[int, int] = (1, 1)):
    #
    # input: ( N, C_in, H_in, W_in )
    # output: ( N, C_out, H_out, W_out )
    #
    # H_out = (H_in + 2 * padding[0] - dilation[0] * ( kernel_size[0] - 1) - 1) / stride[0] + 1
    # W_out = (w_in + 2 * padding[1] - dilation[1] * ( kernel_size[1] - 1) - 1) / stride[1] + 1
    #

    if in_channel != shape[1]:
        exit(f"conv2d in_channel miss-matching [ in_channel in argument: {in_channel} input shape: {shape} ]")

    H_out = (shape[2] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
    W_out = (shape[3] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
    return [shape[0], out_channel, H_out, W_out]


def conv1d(shape: list[int], in_channel: int = None, out_channel: int = None,
           kernel: int = None,
           padding: int = 0, stride: int = 1,
           dilation: int = 1):
    if in_channel != shape[1]:
        exit(f"conv2d in_channel miss-matching [ in_channel in argument: {in_channel} input shape: {shape} ]")
    print_layer(layer_name="conv1d", shape=shape)
    W_out = (shape[2] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1

    return [shape[0], out_channel, W_out]

def GELU(shape: list[int]):
    print_layer(layer_name="GLUE", shape=shape)
    return shape

def softmax(dim, shape):
    print_layer(layer_name=f"softmax dim={dim}",line="method",  shape=shape)
    return shape

def dropout(num, shape):
    print_layer(layer_name=f"dropout {num}",line="method",  shape=shape)
    return shape

def batchNorm2d(shape: list[int]):
    print_layer(layer_name="BatchNorm2d", shape=shape)
    return shape

def layerNorm1d(shape: list[int]):
    print_layer(layer_name="layerNorm1d", shape=shape)
    return shape

def batchNorm1d(shape: list[int]):
    print_layer(layer_name="BatchNorm1d", shape=shape)
    return shape
def N_Branch_Jamo_Conv(shape: list[int], in_channel: int = None, out_channel: int = None, number_of_heads: int = 4,
                       kernel: tuple[int, int] = None,
                       padding: tuple[int, int] = (0, 2), stride: tuple[int, int] = (1, 1),
                       dilation: tuple[int, int] = (1, 1)):
    if in_channel % number_of_heads != 0:
        exit(
            f"NB-Jamo-conv in branch-conv in_channel is must be integer. [ args[in_channel, number_of_heads ]=[{in_channel}, {number_of_heads}] branch-conv in_channel={in_channel / number_of_heads}  ]")
    if out_channel % number_of_heads != 0:
        exit(
            f"NB-Jamo-conv in branch-conv out_channel is must be integer. [ args[out_channel, number_of_heads ]=[{out_channel}, {number_of_heads}] branch-conv in_channel={in_channel / number_of_heads}  ]")

    branch_conv_in_channel = int(in_channel / number_of_heads)
    branch_conv_out_channel = int(out_channel / number_of_heads)
    chunk_shape = [shape[0], branch_conv_in_channel, shape[2], shape[3]]
    print_layer(layer_name="N_Branch_Jamo_conv", shape=chunk_shape, shape_type="chunk", status="active")

    shape = conv2d(shape=chunk_shape, in_channel=branch_conv_in_channel, out_channel=branch_conv_out_channel,
                   kernel=kernel, padding=padding,
                   stride=stride, dilation=dilation)
    print_layer(layer_name="N_Branch_Jamo_conv", shape=shape, status="end")
    return shape


def depth_wise_separable_convolution_conv1d(shape: list[int], in_channel: int = None, out_channel: int = None,
                                            kernel: int = None,
                                            padding: int = 0, stride: int = 1,
                                            dilation: int = 1,
                                            group: int = 0):
    if in_channel != shape[1]:
        exit(f"conv2d in_channel miss-matching [ in_channel in argument: {in_channel} input shape: {shape} ]")
    print_layer(layer_name="conv1d", shape=shape)
    W_out = (shape[2] + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    # kernel: 1의 conv1d를 진행함 ( 코드 상 에서는 생략됨 )

    return [shape[0], out_channel, W_out]


def transformer_encoder(shape):
    # 기존 transformer encoder 연산과 동일 하기에 생략
    print_layer(layer_name="transformer encoder", shape=shape, status="end")
    return shape


def jamo_conv_transformer_model():
    # DEFAULT ARGS
    batch_size = 32
    sentence_length = 2 ** 10
    char_dim = 3
    d_model = 256
    # N_Branch_Jamo_conv ARGS
    total_in_channel = 1024
    total_out_channel = 1024
    kernel = (3, 5)

    x = input_layer([batch_size, sentence_length, char_dim])
    x = embedding(x, d_model)

    x_chunk_arr = N_Branch_Jamo_Conv(shape=x, in_channel=total_in_channel, out_channel=total_out_channel, kernel=kernel)

    # chunk 연산
    x_chunk_arr = batchNorm2d(x_chunk_arr)
    x_chunk_arr = GELU(x_chunk_arr)
    x_chunk_arr[1] = x_chunk_arr[1] * 4

    x = squeeze(x_chunk_arr, -2)
    x = fc(x, 512)
    x = transformer_encoder(x)

    # fc
    x1 = fc(x, 256)
    x1 = GELU(x1)

    x1 = fc(x1, 128)
    x1 = dropout(0.1, x1)

    x1 = fc(x1, 1)
    x1 = layerNorm1d(x1)  # (b, 1024, 1)

    # attention pooling
    score = softmax(shape=x1, dim=1)  # (b, 1024, 1)
    x2 = torch_sum(x, dim=2)  # (b, 1024)   x * score
    x2 = unsqueeze(x2, -1)  # (b, 1024, 1)
    x = [batch_size, 1024] # residual  x2 + x1
    x = layerNorm1d(x)
    x = GELU(x) # (b, 1024)


    # return x

    #chunk_arr = torch.split(x)
    #for i in range(number_of_heads):
    #    x1 = fc(chunk_arr[i], 1).squeeze(-1) # (b, 1024, 64) > (b, 1024, 1)
    #    x1 = softmax(x1,1)
    #    chunk_arr[i] = torch_sum(chunk_arr[i]*x1, dim=1) # (b, 64)
    #   torch.concat(chunk_arr, 1)
    # 헤더에 맞게 조정 가능 상태

def search_conv_parameter(kernel: int, before_out: int, range_operator_group: Callable):
    arr = []
    for k in range(kernel):
        if k % 2 != 0:
            continue
        for p in range(0, int(k / 2)):
            for s in range(1,2):
                out = (before_out + 2 * p - 1 * (k - 1) - 1) / s + 1
                if out.is_integer() is False:
                    continue
                if range_operator_group(out) == False:
                    continue
                arr.append([k, p, s, out])
    return arr


if __name__ == "__main__":
    jamo_conv_transformer_model()
    # params_set_array = search_conv_parameter(16, 512, lambda x: 512 > x)
    # if params_set_array.__len__() > 0:
    #     for params_set in params_set_array:
    #         print(
    #             f"first conv = k: {params_set[0]}, p: {params_set[1]}, s: {params_set[2]}, output size: {params_set[3]}")


    # K = 4 or 6
    #
    # # padding < k && stride < k
    # for k in range(16):
    #     for p in range(k):
    #         for s in range(1, k):
    #             out = (512 + 2 * p - 1 * (k - 1) - 1) / s + 1
    #             if out.is_integer() is False:
    #                 continue
    #             if k - s != p:
    #                 continue
    #             if out > 350 or out < 250:
    #                 continue
    #             for sk in range(k):
    #                 for sp in range(sk):
    #                     for ss in range(1, sk):
    #                         sout = (out + 2 * sp - 1 * (sk - 1) - 1) / ss + 1
    #                         if sout.is_integer():
    #                             if sk - ss == sp:
    #                                 if sout < 200:
    #                                     print(f"first conv = k: {k}, p: {p}, s: {s}, output size: {out}")
    #                                     print(f"\t sencond conv = k: {sk}, p: {sp}, s: {ss}, output size: {sout}")

    # k = n 일때 모든 원소를 동일하게 적용될 경우
    # k - s = p

    # x = torch.zeros(size=(32, 2 ** 10), dtype=torch.int)
    # embedding = torch.nn.Embedding(num_embeddings=2 ** 10, embedding_dim=512, dtype=torch.float32, sparse=True)
    # # conv2d = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=8, padding=3, stride=2)
    # conv1d_1 = torch.nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=8, padding=3, stride=2)
    # linear_1 = torch.nn.Linear(256, 1)
    # softmax_1 = torch.nn.Softmax(1)
    # # conv1d_2 = torch.nn.Conv1d(in_channels=512, out_channels=512, kernel_size=128, padding=0, stride=1, groups=512)
    # # conv1d_2_1 = torch.nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
    # x = embedding(x)
    # x = conv1d_1(x)
    # print(x.shape)
    # x1 = linear_1(x)
    # print(x1.shape)
    # x1 = softmax_1(x1)
    # print(x1.shape)
    # x = x * x1
    # print(x.shape)
    # x = torch.sum(x * x1, dim=1)
    # print(x.shape)
