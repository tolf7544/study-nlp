import numpy as np


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)  # 오버플로 방지

    sum_exp_x = np.sum(exp_x)

    y:np.ndarray = exp_x / sum_exp_x
    return y


def sigmoid(x):

    exp_x = np.exp(-x)  # 오버플로 방지

    x = 1 / (1 + exp_x)
    return x


def test_2_active_function(x): # active function 으로는 부적합
    x1 = np.max(x)
    x2 = np.exp(x - x1)
    x3 = 1 / (1 + x2)
    return (1 - x3)/0.5

if __name__ == '__main__':
    test_array = np.array([-0.54, -0.24, -0.1, -0.035, -0.677])

    # test_1_softmax_sigmoid(test_array)
    a = sigmoid(test_array)
    b = softmax(test_array)
    c = test_2_active_function(test_array)

    print(a)
    print(b)
    print(c)
