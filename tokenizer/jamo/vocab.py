import json
import os
from typing import Optional

from corpus.type import TypePath
from util import printl


class Vocab():
    vocab_path: TypePath
    vocab: Optional[object]

    def __init__(self, path: TypePath = "./load_test_vocab.json"):
        self.vocab_path = path
        self.vocab = None

    def __check_vocab_exist(self) -> bool:
        r"""path에 vocab이 존재할 시 true를 리턴하며 오류 발생 또는 아닐 시 출력값과 False를 리턴합니다."""
        if os.path.exists(self.vocab_path):
            return True
        else:
            return False

    def load_vocab(self) -> None:
        r"""vocab을 load하여 self.vocab에 저장합니다."""
        # https://www.geeksforgeeks.org/convert-list-of-tuples-to-json-python/
        
        if self.__check_vocab_exist() is True:
            if self.vocab_path[-3:] == "txt":
                with open(self.vocab_path, "r", encoding="utf-8") as file:
                    vocab_array = file.readlines()
                    self.vocab = [{f"{i}": token.strip()} for (i, token) in list(enumerate(vocab_array))]
            elif self.vocab_path[-4:] == "json":
                with open(self.vocab_path, "r", encoding="utf-8") as file:
                    self.vocab = json.load(file)
            else:
                exit(
                    f"vocab must be .txt or .json file.\ninput path: {self.vocab_path}")
        else:
            printl(f"vocab is not exist in next location ( this message is not error )\ninput path: {self.vocab_path}")
        print(self.vocab)

    def get_vocab(self) -> Optional[object]:
        if self.vocab == None:
            exit(
                "vocab is not exist.\n you can try this solution.\n1. load pre-trained vocab\n input corpus for making vocab")
        else:
            return self.vocab
    def index_2_key(self):
        ...
    def key_2_index(self):
        ...