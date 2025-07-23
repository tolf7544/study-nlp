import torch
import torch.nn as nn


class ClassifierHeader(nn.Module):
    def __init__(self,  model: nn.Module,class_num: int, hidden_layer: int, dropout=0.1):
        super(ClassifierHeader, self).__init__()
        self.model = model
        self.hidden_layer = hidden_layer

        self.classifier = nn.Linear(hidden_layer, class_num)
        self.dropout = nn.Dropout(dropout)


    def forward(self, *args):
        x = self.model(*args)
        x = self.dropout(x)
        x = self.classifier(x).squeeze(1)
        return x