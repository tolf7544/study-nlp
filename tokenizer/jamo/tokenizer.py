from typing_extensions import Unpack

import json
import os
from typing import Optional, Union, Any

import numpy as np
from jamo import j2hcj, h2j, j2h, is_jamo

from corpus.data_normalizers import DataNormalizers
from corpus.type import JamoEncodingType, Path, NormalizerOption
from tokenizer.jamo.decoder import JamoTokenizerDecoder
from tokenizer.jamo.encoder import JamoTokenizerEncoder
from tokenizer.jamo.vocab import Vocab
from tokenizer.type import JamoMatrix, \
    JamoMatrixDataset, DefaultSpecialToken, JamoCharArray, JamoCharArrayDataset
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
            self.debug.opt_debug_print(
                f"vocab is not exist in next location ( this message is not error )\ninput path: {self.vocab_path}")
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
    maximum_sentence_length: int
    encoding_type: JamoEncodingType
    truncation: bool = False
    padding: bool = False
    normalizer_queue: list[NormalizerOption]
    r"""입력된 순서대로 정규화를 진행하며 각 과정은 고유한 특성(중복x)을 갖는다. first input first out"""

    vocab: Vocab
    data_normalizer: DataNormalizers
    debug: Debug

    __jamo_tokenizer_encoder: JamoTokenizerEncoder
    __jamo_tokenizer_decoder: JamoTokenizerDecoder

    def __init__(self,
                 vocab_path: Path = "./vocab.json",
                 special_token: list[str] = None,
                 normalizer: list[Unpack[NormalizerOption]] = None,
                 encoding_type: JamoEncodingType = JamoEncodingType.JAMO_VECTOR,
                 ):
        self.vocab_path = vocab_path
        self.vocab = Vocab(LoadVocab(vocab_path).load_file(), special_token)

        self.normalizer_queue = normalizer
        self.data_normalizer = DataNormalizers()

        self.encoding_type = encoding_type
        self.__jamo_tokenizer_encoder = JamoTokenizerEncoder(self.vocab)
        self.__jamo_tokenizer_decoder = JamoTokenizerDecoder(self.vocab)

        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))
        self.sentence_length = 200
        self.maximum_sentence_length = 0
    def __h2hcj(self, jamo_char) -> JamoCharArray:
        r"""
        한글 문장 > h2j(자소 단위 정규화) > j2hcj(호환 자모 정규화)
        """
        result = []

        for char in jamo_char:
            char = list(j2hcj(h2j(char)))
            if char.__len__() < 3:
                for i in range(3 - char.__len__()):
                    char.append(DefaultSpecialToken._member_names_[0])  # padding은 0번째에 위치함
            result.append(char)

        if jamo_char.__len__() > self.maximum_sentence_length:
            self.maximum_sentence_length= jamo_char.__len__()
        return result

    def __jamo_string_2_h(self, jamo_array: JamoCharArray, is_hide_token: bool) -> str:
        r"""( 3, jamo ) shape 형태 백터 한글 변형"""
        char_jamo: np.ndarray[str, str, str]
        result: str = ""

        for char_jamo in jamo_array:
            if char_jamo[0].__len__() == 1 and char_jamo[1].__len__() == 1  and is_jamo(char_jamo[0]):  # 초성 중성이 존재할 시에만 j2h 메서드 사용
                for i in range(char_jamo.__len__()):
                    if char_jamo[i] == DefaultSpecialToken._member_names_[0]:
                        char_jamo[i] = ""
                result += j2h(*char_jamo)
            else:
                if not is_hide_token or not char_jamo[0] in DefaultSpecialToken._member_names_:
                    result += char_jamo[0]
        return result

    def add_jamo_from_corpus(self, corpus: list[str]):
        r"""add jamo to vocab"""

        for content in corpus:
            content = self.data_normalizer.filtering_normalize(content)
            content = j2hcj(h2j(content))
            for char in list(set(content)):
                self.vocab.add(char)  # has()문이 내장되어 있음.

        self.debug.opt_debug_print("successfully added")
        self.debug.opt_debug_print(self.vocab)

        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, encoding="utf-8", mode="w") as f:
                f.write(json.dumps(self.vocab, ensure_ascii=False))
        else:
            with open("./tokenizer/data/vocab.json", encoding="utf-8", mode="w") as f:
                f.write(json.dumps(self.vocab, ensure_ascii=False))

    def tokenize(self, sentences: Union[Optional[str], list[str]]) \
            -> Optional[list[list[str]]]:

        if not isinstance(sentences, list) and not isinstance(sentences, str):
            self.debug.unexpected_error("sentences args type must be str or list[str]")
            return None

        if isinstance(sentences, str):
            sentences = [sentences]

        self.data_normalizer.set_corpus(sentences)
        sentences = self.data_normalizer.compute_normalize()
        sentences = [self.__h2hcj(sentence) for sentence in sentences]

        return sentences

    def merge_token(self, sentences: Union[Optional[JamoCharArrayDataset], JamoCharArray], is_hide_token: bool = True):
        if len(sentences[0]) != 3 and len(sentences[0][0]) != 3:
            self.debug.unexpected_error(
                "sentences argument must be JamoVector or JamoVectorDataset. use this method after processing \"JamoTokenizer.tokenize()\" ")
            return None
        print("sentences")
        if len(sentences[0]) == 3 and not isinstance(sentences[0][0], np.ndarray):
            sentences = [sentences]

        for i, single_sentence in enumerate(sentences):
            sentences[i] = self.__jamo_string_2_h(JamoCharArray(single_sentence), is_hide_token)

        return sentences

    def encode(self,
               sentences: Union[Optional[JamoCharArray], JamoCharArrayDataset],
               sentence_length: int = 200,
               padding: bool = False,
               truncation: bool = False) -> Optional[JamoMatrixDataset]:

        if len(sentences[0]) != 3 and len(sentences[0][0]) != 3:
            self.debug.unexpected_error(
                "sentences argument must be JamoVector or JamoVectorDataset. use this method after processing \"JamoTokenizer.tokenize()\" ")
            return None

        if len(sentences[0]) == 3 and not isinstance(sentences[0][0], list):
            sentences = [sentences]

        if not truncation:
            sentence_length = self.maximum_sentence_length

        encoded_vector = np.zeros((len(sentences), sentence_length, 3, self.vocab.length), dtype=np.float64)
        for i in range(sentences.__len__()):
            encoded_vector[i] = self.__jamo_tokenizer_encoder.encode_one_hot_vector_matrix(
                jamo_char_array=sentences[i],
                sentence_length=sentence_length,
                padding=padding,
                truncation=truncation)
        return  encoded_vector

    def decode(self,
               sentences: Union[Optional[JamoMatrix], JamoMatrixDataset]) \
            -> Optional[JamoMatrixDataset]:

        if ((sentences.shape[1] != 3 and sentences.shape[2] != 3)
                or (sentences.shape[2] != self.vocab.length and sentences.shape[3] != self.vocab.length)):
            self.debug.unexpected_error(
                "input data type is JamoMatrix or JamoMatrixDataset,  use this method after processing \"JamoTokenizer.encode()\" or if you same shape of arguments, change type force ")
            return None
        if sentences.shape[1] == 3 and sentences.shape[2] == self.vocab.length:
            sentences = [sentences]
        decoded_vector = np.zeros((sentences.shape[0],), dtype=np.object_)
        for i in range(sentences.__len__()):
            decoded_vector[i] = self.__jamo_tokenizer_decoder.decode_one_hot_vector_matrix(sentences[i])
        return decoded_vector

    def __call__(self,
                 sentences: Union[np.ndarray, Union[str, list[str]]],
                 sentence_length: int = 200,
                 padding: bool = True,
                 truncation: bool = True,
                 normalizer: list[Unpack[NormalizerOption]] = None,
                 encoding_type: JamoEncodingType = JamoEncodingType.JAMO_VECTOR):

        self.sentence_length = sentence_length
        self.padding = padding
        self.truncation = truncation
        self.normalizer_queue = normalizer
        self.encoding_type = encoding_type

        x = self.tokenize(sentences)
        x = self.encode(x, self.sentence_length, self.padding, self.truncation)
        return x
