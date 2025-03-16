import re
from os import error

import numpy as np
import torch.cuda
from sympy import false
from transformers import PreTrainedTokenizerFast

from corpus.custom_normalizers import SentenceNormalizers
from corpus.data_loader import CorpusLoader
from tokenizer.pre_training import pre_training_tokenizer, load_data


def test_tokenizer():
    result = load_data("read")
    # tokenizer = pre_training_tokenizer(result)

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/vocab/tokenizer.json",
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    while True:
        data = input("\n입력:")
        print(wrapped_tokenizer.encode(data))
        print(wrapped_tokenizer.tokenize(data))


def test_Corpus_normalizers():
    contents = [
        "개웃기네ㅋㅋㅋㅋㅋㅋㅋ",
        "와아아아아앙 우오오오오아앙",
        "이거 개쩜 https://arxiv.org/pdf/2004.05150 진짜 미쳤다",
        "실화냐........",
        "와우와우와우와우",
        "어쩔어쩔어쩔어쩔",
        "응아니야응아니야응아니야",
        "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        "웃겨ㅕㅕㅕㅕㅕㅕㅕㅕㅕㅕ",
        "빼에에에에에에에에에에엑",
        "와아아아아아아아아아아아",
        "응ㅋ어쩔어쩔어쩔어쩔어쩔",
        "어쩔티비어쩔티비어쩔티비",
        "ㅅㄱㄹㅅㄱㄹㅅㄱㄹ"
    ]
    corpus = SentenceNormalizers()
    print(corpus.normalizer(contents).contents)
    print(corpus.get_error_contents())
    # for sentence in contents:
    #     print(corpus.remove_repetition_char(content=sentence))
    #     print(corpus.remove_url(content=sentence))
def console_log(input: str):
    print(input)
if __name__ == '__main__':
    # print("cuda status: ", torch.cuda.is_available())
    # test_Corpus_normalizers()

    loader = CorpusLoader()
    contents = loader.open_file("corpus/test_data/type1.csv", "type1.csv").contents.map()
    print(contents)
    #print(contents.all)

