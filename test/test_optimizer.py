from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

from transformers import AutoImageProcessor, ResNetForImageClassification

if __name__ == '__main__':
    dataset = load_dataset("huggingface/cats-image")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

