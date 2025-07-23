import math
from datetime import datetime
from math import floor

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision

from model.cosine_annealiong_with_warmup import CosineAnnealingWarmupRestarts
from model.jamo_cnn_transformer import JamoCNNTransformerModel
from tokenizer.jamo_tokenizer import JamoTokenizer

tokenizer = JamoTokenizer(vocab_path="vocab.json")

post_processing_metadata = ["down_scaling_mask", "attention_mask", "encode", "label"]

model_name = "JamoCNNTransformerModel"
sequence_len = 512
embedding_len = 256
cnn_num_heads = 4
d_model = 512
d_pwff = 2048

batch = 64
device = "cuda:0"
num_class = 1
warmup_steps = 4000
early_stopping_limit = 5


dataset_size = 0
warmup_step_ratio = 8 / 9
step = 1000

# 0: negative
# 1: positive
def post_processing(row: dict):
    row = row["text"].split("\t")
    sentence = tokenizer.normalize(row[1])
    tokenized = tokenizer.tokenize(sentence, length=sequence_len, padding=True, truncation=True)
    encoded = tokenizer.encode(tokenized, return_attention_mask=True)

    return {
        "down_scaling_mask": torch.tensor(encoded["down_scaling_mask"], dtype=torch.float32),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.float32),
        "encode": torch.tensor(encoded["encode"], dtype=torch.int),
        "label": torch.tensor(int(row[2]), dtype=torch.float32)
    }


#
# https://bluecolorsky.tistory.com/82
#

# #
# 텐서보드로 학습률 실시간 표기 추가
# #

def calculate_warmup_step():
    return floor(dataset_size / batch * warmup_step_ratio)

def calculate_epoch_step():
    return round(dataset_size / batch)

def compute_metrics(x_pred, y_target):
    accuracy = BinaryAccuracy().to(device)
    f1_score = BinaryF1Score().to(device)
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    return (
        f1_score(x_pred, y_target),
        accuracy(x_pred, y_target),
        recall(x_pred, y_target),
        precision(x_pred, y_target)
    )


if __name__ == '__main__':
    data_files = {"train": "train.txt", "progress_test": "progress_test.txt"}
    dataset = load_dataset("./corpus", data_files=data_files, split="train[1:50000]")
    dataset = dataset.map(post_processing, remove_columns="text", num_proc=20)
    dataset.set_format(type="torch", columns=post_processing_metadata)
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(0.3)

    train_dataset = DataLoader(
        dataset["train"],
        batch_size=batch,
    )

    eval_dataset = DataLoader(
        dataset["test"],
        batch_size=batch
    )

    model = JamoCNNTransformerModel(tokenizer.vocab.length(), sequence_len, embedding_len, d_model, d_pwff,
                                    cnn_num_heads, device,
                                    num_layers=6, num_class=num_class).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps= calculate_epoch_step(),
        cycle_mult=1.0,
        max_lr=1e-1,
        min_lr=1e-4,
        warmup_steps=calculate_warmup_step(),
        gamma=1.0)

    loss_fn = torch.nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter('runs/model_{}_{}'.format(model_name, timestamp))

    over_loss_count = 0
    epoch_number = 0

    EPOCHS = 99999

    best_vloss = 1_000_000.


    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_dataset):
            print('{} / {}'.format(i + 1, len(train_dataset)))
            # Every data instance is an input + label pair

            down_scaling_mask = data["down_scaling_mask"].cuda()
            attention_mask = data["attention_mask"].cuda()
            inputs = data["encode"].cuda()
            labels = data["label"].cuda()

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs, attention_mask, down_scaling_mask)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward(retain_graph=True)

            # Adjust learning weights
            optimizer.step()
            scheduler.step()

            # Gather data and report
            running_loss += loss.item()
            if i % step == step - 1:
                last_loss = running_loss / step  # loss per batch
                f1_score, accuracy, recall, precision = compute_metrics(outputs, labels)
                print('  batch {} loss: {} accuracy {} f1_score {} recall {} precision {}'.format(i + 1, last_loss,
                                                                                                  accuracy, f1_score,
                                                                                                  recall, precision))

                tb_x = epoch_index * len(train_dataset) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('f1_score/train', f1_score, tb_x)
                tb_writer.add_scalar('accuracy/train', accuracy, tb_x)
                tb_writer.add_scalar('recall/train', recall, tb_x)
                tb_writer.add_scalar('precision/train', precision, tb_x)
                running_loss = 0.

        return last_loss


    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(epoch_number)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(eval_dataset):
                print('{} / {}'.format(i + 1, len(eval_dataset)))
                v_down_scaling_mask = vdata["down_scaling_mask"].cuda()
                v_attention_mask = vdata["attention_mask"].cuda()
                v_inputs = vdata["encode"].cuda()
                v_labels = vdata["label"].cuda()

                voutputs = model(v_inputs, v_down_scaling_mask, v_attention_mask)
                vloss = loss_fn(voutputs, v_labels)
                running_vloss += vloss

                f1_score, accuracy, recall, precision = compute_metrics(voutputs, v_labels)
                print(' accuracy {} f1_score {} recall {} precision {}'.format(accuracy, f1_score, recall, precision))
                tb_x = epoch * len(eval_dataset) + i + 1
                tb_writer.add_scalar('f1_score/valid', f1_score, tb_x)
                tb_writer.add_scalar('accuracy/valid', accuracy, tb_x)
                tb_writer.add_scalar('recall/valid', recall, tb_x)
                tb_writer.add_scalar('precision/valid', precision, tb_x)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars('Training vs. Validation Loss',
                              {'Training': avg_loss, 'Validation': avg_vloss},
                              epoch_number + 1)

        tb_writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            over_loss_count = 0
            best_vloss = avg_vloss
            model_path = './model/{}_{}_{}'.format(model_name, timestamp, epoch_number)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, model_path)
        else:
            over_loss_count += 1
            if over_loss_count > early_stopping_limit:
                print(f'Train Early Stopping after {epoch} epochs.')
                break

        epoch_number += 1
