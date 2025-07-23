import torch
from torch import nn

def test_1():
    parameter_1 = torch.tensor(50.0, dtype=torch.float32)
    parameter_2 = torch.tensor(25.0, dtype=torch.float32)
    parameter_3 = torch.tensor(0.0, dtype=torch.float32)
    parameter_4 = torch.tensor(-25.0, dtype=torch.float32)

    parameter_list = [parameter_1,parameter_2,parameter_3,parameter_4]
    gelu = nn.GELU()
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    softplus = nn.Softplus()
    softmax = nn.Softmax(dim=0)
    # print(gelu(parameter))
    # print(relu(parameter))
    # print(sigmoid(parameter))

    for param_data in parameter_list:
        print(f"----- {param_data.data} -----")
        print(f"softmax {softmax(param_data)}")
        print(f"softplus {softplus(param_data)}")
        print(f"gelu {gelu(param_data)}")
        print("")

if __name__ == '__main__':
    test_1()