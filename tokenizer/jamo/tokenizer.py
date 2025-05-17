from typing_extensions import Unpack

import json
import os
from typing import Optional, Union

import numpy as np
from jamo import j2hcj, h2j, j2h

from corpus.data_normalizers import DataNormalizers
from corpus.type import JamoEncodingType, Path, NormalizerOption
from tokenizer.jamo.vocab import Vocab
from tokenizer.type import JamoMatrix, \
    JamoMatrixDataset, CustomSpecialToken, DefaultSpecialToken, JamoCharArray, JamoCharArrayDataset
from util import Debug


class LoadVocab:
    vocab_path: Path
    debug: Debug
    def __init__(self, _vocab_path: Optional[Path] = None):
        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))
        self.vocab_path = _vocab_path


    def __check_vocab_exist(self) -> bool:
        r"""path에 vocab이 존재할 시 true를 리턴하며 오류 발생 또는 아닐 시 출력값과 False를 리턴합니다."""
        if os.path.exists(self.vocab_path):
            return True
        else:
            return False

    def load_file(self) -> dict:
        r"""vocab을 load하여 self.vocab에 저장합니다."""
        # https://www.geeksforgeeks.org/convert-list-of-tuples-to-json-python/

        _key_vocab: dict = {}

        if self.__check_vocab_exist() is False:
            self.debug.opt_debug_print(f"vocab is not exist in next location ( this message is not error )\ninput path: {self.vocab_path}")
            return {}

        with open(self.vocab_path, "r", encoding="utf-8") as file:
            if self.vocab_path[-3:] == "txt":
                vocab_array = file.readlines()
                _key_vocab = {i: token.strip() for (i, token) in list(enumerate(vocab_array))}
            elif self.vocab_path[-4:] == "json":
                _key_vocab = json.load(file)
            else:
                self.debug.unexpected_error(
                    f"vocab must be .txt or .json file.\ninput path: {self.vocab_path}")

            return _key_vocab


class JamoTokenizer:
    vocab_path: Path
    sentence_length: int
    encoding_type: JamoEncodingType
    truncation: bool = False
    padding: bool = False
    normalizer_queue: list[NormalizerOption]
    r"""입력된 순서대로 정규화를 진행하며 각 과정은 고유한 특성(중복x)을 갖는다. first input first out"""

    vocab: Vocab
    dataNormalizer: DataNormalizers
    debug: Debug
    def __init__(self,
                 vocab_path: Path = "./vocab.json",
                 sentence_length: int = 200,
                 padding: bool = False,
                 truncation: bool = False,
                 special_token: list[CustomSpecialToken] = None,
                 normalizer: list[Unpack[NormalizerOption]] = None,
                 encoding_type: JamoEncodingType = JamoEncodingType.JAMO_VECTOR,
                 ):
        self.vocab_path = vocab_path
        self.sentence_length = sentence_length

        self.padding = padding
        self.truncation = truncation

        self.vocab = Vocab(LoadVocab(vocab_path).load_file(), special_token)

        self.normalizer_queue = normalizer
        self.encoding_type = encoding_type

        self.dataNormalizer = DataNormalizers()
        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))

    def __h2hcj(self, content) -> JamoCharArray:
        r"""
        한글 문장 > h2j(자소 단위 정규화) > j2hcj(호환 자모 정규화)
        """
        result: list = []
        for char in content:
            char = np.array(j2hcj(h2j(char)))
            if char.__len__() < 3:
                for i in range(3 - char.__len__()):
                    np.append(char, self.vocab.get(DefaultSpecialToken._member_names_[0])) # padding은 0번째에 위치함
            np.append(result, char)
        return JamoCharArray(result)

    def __jamo_string_2_h(self, jamo_array: JamoCharArray) -> str:
        r"""( 3, jamo ) shape 형태 백터 한글 변형"""
        result:str = ""
        for char_jamo in jamo_array:
            print(*char_jamo)
            if char_jamo[0].__len__() == 1 and char_jamo[1].__len__() == 1: # 초성 중성이 존재할 시에만 j2h 메서드 사용
                result += j2h(*char_jamo)
            else:
                result += char_jamo[0]

        return result

    def add_jamo_from_corpus(self, corpus: list[str]):
        r"""add jamo to vocab"""

        for content in corpus:
            content = self.dataNormalizer.filtering_normalize(content)
            content = j2hcj(h2j(content))
            for char in list(set(content)):
                self.vocab.add(char) # has()문이 내장되어 있음.

        self.debug.opt_debug_print("successfully added")
        self.debug.opt_debug_print(self.vocab)

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, encoding="utf-8", mode="w") as f:
                f.write(json.dumps(self.vocab, ensure_ascii= False))
        else:
            with open("./tokenizer/data/vocab.json", encoding="utf-8", mode="w") as f:
                f.write(json.dumps(self.vocab, ensure_ascii= False))


    def __encoding_one_hot_vector(self):
        ...
    def __create_zero_vector(self):
        ...


    def tokenize(self, sentence: Union[Optional[str], list[str]]) -> Union[Optional[JamoCharArray], JamoCharArrayDataset]:
        if isinstance(sentence, JamoCharArrayDataset):
            self.dataNormalizer.set_corpus(sentence)
            sentence = self.dataNormalizer.compute_normalize()
        else:
            sentence = self.dataNormalizer.filtering_normalize(str(sentence))
        sentence = self.__h2hcj(sentence)
        return sentence
    
    def merge_token(self, sentence: Union[Optional[JamoCharArrayDataset], JamoCharArray]):
        if isinstance(sentence, JamoCharArray):
            sentence = self.__jamo_string_2_h(sentence)
        else:
            for i, single_sentence in enumerate(sentence):
                sentence[i] = self.__jamo_string_2_h(JamoCharArray(single_sentence))
        return sentence

    def encode(self, sentence: Union[Optional[JamoMatrix],JamoMatrixDataset]):
        if isinstance(sentence, JamoMatrix):
            ...
        elif isinstance(sentence, JamoMatrixDataset):
            ...
        else:
            self.debug.unexpected_error("sentence argument must be JamoVector or JamoVectorDataset. you must use method \"JamoTokenizer.tokenize\" before sentence encoding.")

        # unit_set = set()
        # unit_key_pair = []
        #
        # for word in unit_set:
        #     self.vocab.token_2_key(word)


