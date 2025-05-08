import json
import os
from typing import Optional
from corpus.data_normalizers import DataNormalizers
from corpus.type import JamoEncodingType, TypePath, NormalizationMethod
from tokenizer.jamo.vocab import Vocab
from tokenizer.type import ZeroVector
from util import Debug


class LoadVocab():
    vocab_path: TypePath
    debug: Debug
    def __init__(self, _vocab_path: Optional[TypePath] = None):
        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))
        self.vocab_path = _vocab_path

    def __check_vocab_exist(self) -> bool:
        r"""path에 vocab이 존재할 시 true를 리턴하며 오류 발생 또는 아닐 시 출력값과 False를 리턴합니다."""
        if os.path.exists(self.vocab_path):
            return True
        else:
            return False

    def load_file(self) -> Vocab:
        r"""vocab을 load하여 self.vocab에 저장합니다."""
        # https://www.geeksforgeeks.org/convert-list-of-tuples-to-json-python/

        _key_vocab: dict = {}

        if self.__check_vocab_exist() is False:
            self.debug.opt_debug_print(f"vocab is not exist in next location ( this message is not error )\ninput path: {self.vocab_path}")
            return Vocab({})

        with open(self.vocab_path, "r", encoding="utf-8") as file:
            if self.vocab_path[-3:] == "txt":
                vocab_array = file.readlines()
                _key_vocab = {i: token.strip() for (i, token) in list(enumerate(vocab_array))}
            elif self.vocab_path[-4:] == "json":
                _key_vocab = json.load(file)
            else:
                self.debug.unexpected_error(
                    f"vocab must be .txt or .json file.\ninput path: {self.vocab_path}")

            return Vocab(_key_vocab)


class JamoTokenizer():
    vocab_path: TypePath
    sentence_length: int
    encoding_type: JamoEncodingType

    truncation: bool = False
    padding: bool = False
    normalizer_queue: list[NormalizationMethod]
    r"""입력된 순서대로 정규화를 진행하며 각 과정은 고유한 특성(중복x)을 갖는다. first input first out"""

    vocab: Vocab
    dataNormalizer: DataNormalizers
    debug: Debug
    def __init__(self,
                 vocab_path: TypePath = None,
                 sentence_length: int = 200,
                 padding: bool = False,
                 truncation: bool = False,
                 normalizer: list[NormalizationMethod] = None,
                 encoding_type: JamoEncodingType = JamoEncodingType.JAMO_VECTOR,
                 ):
        self.padding = padding
        self.truncation = truncation
        self.sentence_length = sentence_length
        self.normalizer_queue = normalizer
        self.encoding_type = encoding_type
        self.dataNormalizer = DataNormalizers()
        self.vocab = LoadVocab(vocab_path).load_file()

    def add_jamo_from_corpus(self, corpus: list[str]):
        r"""add jamo to vocab"""

        for content in corpus:
            content = self.dataNormalizer.filtering_normalize(content)
            for char in list(set(content)):
                self.vocab.add(char) # has()문이 내장되어 있음.

        self.debug.opt_debug_print("successfully added")
        self.debug.opt_debug_print(self.vocab)

    def __zero_sparse_vector(self):
        ... 

    def tokenize(self, sentence: str):
        self.dataNormalizer.filtering_normalize(sentence=sentence)

    def encode(self, sentence: str):
        unit_set = set(sentence)
        unit_key_pair = []

        for word in unit_set:
            self.vocab.token_2_key(word)


