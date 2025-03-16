import json
import pathlib
from typing import Optional

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

data_path = {
    "read": "E:\dataset\sample corpus\Sample\labeled\BOHE210000025292.json",

    "processed-test": "E:\dataset\sample corpus\Sample\\txt\\test.json",
    "processed-train_data": "E:\dataset\sample corpus\Sample\\txt\\train.json",
    "processed-eval_data": "E:\dataset\sample corpus\Sample\\txt\\eval.json",
}


def pre_tkz_debug(str):
    print("\n-------------------------------------")
    print("debug log:", str)
    print("-------------------------------------")

    #데이터 로드 및 저장 함수


def load_data(path_option):
    pylist = []

    if path_option == "read":
        pre_tkz_debug("corpus corpus loading start")
        with open(data_path["read"], "r", encoding="utf-8") as json_data:
            json_data = json.load(json_data)
            pylist = [cotents["content"][0]["sentence"] for cotents in json_data["named_entity"]]
            pre_tkz_debug("pylist created.")

            pre_tkz_debug("corpus corpus loaded")
            with open(data_path["processed-test"], "w", encoding="utf-8") as file:
                file.write(json.dumps(pylist))
                pre_tkz_debug("file saved.")
    elif path_option == "processed-test":
        pre_tkz_debug("processed-test corpus load start")
        with open(data_path["processed-test"], "r", encoding="utf-8") as json_data:
            json_data = json.load(json_data)
            pylist = json_data
            pre_tkz_debug("processed-test corpus loaded")
    elif path_option == "processed-train_data":
        ...
    else:
        ...
    pre_tkz_debug("load_data_finished")

    return pylist

def pre_training_tokenizer(pylist):
    # load WordPiece
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.BertNormalizer()]
    )

    # prepairing pre-training
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(pylist, trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.save("./tokenizer/corpus/tokenizer.json")

    return tokenizer
