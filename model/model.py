import torch
import torch.nn as nn
from torch import Tensor

from caculate.shape import softmax
from model.attention_pooling import AttentionPooling


class JamoCNNTransformerModel(nn.Module):
    def __init__(self,sequence_length: int, embedding_dim: int, cnn_number_of_heads: int):
        super(JamoCNNTransformerModel, self).__init__()
        self.embedding = nn.Embedding(sequence_length, embedding_dim, dtype=torch.float32, sparse=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.gelu = nn.GELU

        self.dropout1 = nn.Dropout1d(0.1)
        self.layer_norm= nn.LayerNorm(embedding_dim)

        self.attention_pooling = AttentionPooling(embedding_dim=embedding_dim)


    def forward(self, x):


        # #
        # dimension reduction progress
        # #
        x = self.fc1(x) # (batch_size, 1024, 512) -> (batch_size, 1024, 256)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x) # (batch_size, 1024, 128) -> (batch_size, 1024, 1)
        x = self.layer_norm(x)

        x = self.attention_pooling(x) # (batch_size, 1024, 1) -> (batch_size, 1024)

        ###
        #   header 위치
        ###