import torch

from model.classifier import ClassifierHeader
from model.jamo_cnn_transformer import JamoCNNTransformerModel
from model.transformer.transformer_encoder import TransformerEncoder
from model.transformer.transformer_encoder_block import TransformerEncoderBlock

def test_1():
    transformer_model = TransformerEncoder(d_model=10, d_pwff=10, device="cpu", sequence_length=10, num_layers=4, num_heads=2, )
    mask = torch.tensor([1,1,1,1,1,0,0,0,0,0], dtype=torch.float32)
    x = torch.randn(1,10,10, dtype=torch.float32)

    x = transformer_model(x, mask)

    print(x.shape)

def test_2():
    sequence_len = 512
    embedding_len = 256
    d_model = 512
    d_pwff = 2048
    cnn_num_heads = 4
    device = "cuda:0"
    model = JamoCNNTransformerModel(sequence_len,embedding_len,d_model,d_pwff, cnn_num_heads, device, num_layers=6).to(device)
    model = ClassifierHeader(model, hidden_layer=sequence_len,class_num=1).to(device)
    # print(sum(p.numel() for p in model.parameters()))

    mask = torch.tensor([1,1,1,1,1,0,0,0,0,0], dtype=torch.float32)
    down_scale_mask = torch.tensor([0,0,0,0,0,1,1,0,0,0], dtype=torch.float32)
    x = torch.zeros((32, 512, 3)).int().to(device)
    _mask = torch.zeros((32, 512), dtype=torch.float32).to(device)
    print(model(x, _mask, _mask).shape)
if __name__ == '__main__':
    test_2()