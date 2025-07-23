import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

from model.jamo_cnn_transformer import JamoCNNTransformerModel

sequence_len = 512
embedding_len = 256
d_model = 512
d_pwff = 2048
cnn_num_heads = 4
device = "cuda:0"
batch = 32
warmup_steps = 40
x_index = []
y_lr = []

def draw_graph(x, y):
    plt.plot(x, y, 'o-', color='red')

    plt.show()

def test_1():
    for step_num in range(1, 100):
        lr = d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
        y_lr.append(lr)
        x_index.append(step_num)

    return [x_index,y_lr]
    # draw_graph(x_index, y_lr)


def test_2():
    model = JamoCNNTransformerModel(sequence_len, embedding_len, d_model, d_pwff, cnn_num_heads, device,
                                    num_layers=6)
    optimizer = torch.optim.AdamW(params=model.parameters(), eps=10e-9, betas=[0.9, 0.98])
    lambda_1 = lambda epoch: d_model ** -0.5 * min((epoch+1) ** -0.5, (epoch+1) * warmup_steps ** -1.5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_1)

    for i in range(1,100):
        scheduler.step()
        y_lr.append(scheduler.get_lr())
        x_index.append(i)

    return [x_index,y_lr]

    # draw_graph(x_index, y_lr)
if __name__ == '__main__':
    print(test_2() == test_1())
    pass

