import torch
import torch.nn as nn

from model.attention_pooling import AttentionPooling
from model.jamo_cnn.jamo_cnn import JamoCNN

from model.transformer.transformer_encoder import TransformerEncoder


class JamoCNNTransformerModel(nn.Module):
    def __init__(self,vocab_size: int, sequence_length: int, embedding_dim: int, d_model: int, d_pwff: int, cnn_num_heads: int,
                 device, num_class: int, num_layers: int = 12, num_heads: int = 8, dropout: int = 0.1,
                 transformer_dropout_scale: int = 0.2, pwff_dropout_scale: int = 0.2):
        super(JamoCNNTransformerModel, self).__init__()

        self.d_model = torch.tensor(d_model,device= device)
        self.embedding_dim = torch.tensor( embedding_dim,device= device)
        self.num_heads = torch.tensor( num_heads, device= device)
        self.num_layers = torch.tensor( num_layers, device= device)

        self.embedding = nn.Embedding(vocab_size+1, embedding_dim, dtype=torch.float32)
        self.transformer_encoder = TransformerEncoder(d_model=d_model, d_pwff=d_pwff, sequence_length=sequence_length,
                                                      device=device, num_layers=num_layers, num_heads=num_heads,
                                                      transformer_dropout_scale=transformer_dropout_scale,
                                                      pwff_dropout_scale=pwff_dropout_scale)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(sequence_length)

        self.attention_pooling = AttentionPooling(embedding_dim=sequence_length, dropout_scale=0.2)
        self.jamo_cnn = JamoCNN(sequence_length=sequence_length, num_heads=cnn_num_heads, dropout=dropout, device=device)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sequence_length, num_class)
        )
    def forward(self, x, pad_mask, down_scaling_mask):  # attention mask 적용 방식을 참조하여 down_scaling_mask 구현해보기
        x = self.embedding(x)
        x = self.jamo_cnn(x, down_scaling_mask)
        # transformer
        x = self.transformer_encoder(x, pad_mask)
        #
        #
        # # #
        # # dimension reduction progress
        # # #
        x = self.fc1(x)  # (batch_size, 1024, 512) -> (batch_size, 1024, 256)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)  # (batch_size, 1024, 128) -> (batch_size, 1024, 1)
        x = x.squeeze(-1)
        x = self.layer_norm(x)

        x = self.attention_pooling(x)  # (batch_size, 1024, 1) -> (batch_size, 1024)
        x = self.classifier(x).squeeze(1)
        return x
