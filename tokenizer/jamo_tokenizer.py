import json
import os
from typing import Union

import numpy as np
from jamo import is_jamo
from tokenizer.jamo_converter import JamoConvertor
from tokenizer.normalizer import Normalizer
from tokenizer.vocab import Vocab
from util.logger import Log
from util.type import JamoSet, SparseList, SparseSet


class JamoTokenizer:
    log = Log(domain_name="JamoTokenizer", save_path=".log", mode="debug", is_save=False, is_display=True)
    normalizer: Normalizer
    jamo_convertor: JamoConvertor
    vocab: Vocab

    def __load_vocab(self, path: str):
        with open(path, encoding="utf-8", mode="r") as f:
            vocab_json = json.load(f)
            special_token_list: list[str] = []


            for item in vocab_json["special_token"].items():
                if item[1]["is_default"] == True:
                    special_token_list.append(item[1]["using_token_name"])
                else:
                    special_token_list.append(item[0])

            self.vocab = Vocab(vocab_json["vocab"], special_token_list,
                               vocab_json["special_token"]["pad"]["using_token_name"],
                               vocab_json["special_token"]["unk"]["using_token_name"],
                               self.log)

    def __init__(self, vocab: Vocab = None, vocab_path: Union[str, os.PathLike[str]] = None):
        if isinstance(vocab, Vocab):
            self.vocab = Vocab(vocab.get_vocab(), list(vocab.special_token().keys()), vocab.pad[0], vocab.unk[0],
                               self.log)

        elif vocab_path:
            self.__load_vocab(vocab_path)
        else:
            self.vocab = Vocab()

        self.normalizer = Normalizer()
        self.jamo_convertor = JamoConvertor()

    def normalize(self, sentence: str) -> str:
        sentence = self.normalizer.remove_control_char(sentence)
        sentence = self.normalizer.normalization_from_combination(sentence)
        sentence = self.normalizer.remove_accent(sentence)
        sentence = self.normalizer.reduce_whitespace(sentence)
        sentence = self.normalizer.reduce_all_repetition_char(sentence)
        return sentence

    def tokenize(self, sentence: str, length: int = 200, padding: bool = False, truncation: bool = False) \
            -> list[JamoSet]:
        r"""
        :param sentence: 문자열
        :param length: 문자열 길이
        :param padding: legnth에 따른 여백 채우기 유무
        :param truncation: length에 따른 overflow된 문자열 제거 유무
        :return: list[list[str,str,str]]
        """
        if sentence.__len__() > length and truncation is True:
            sentence = sentence[:length]
        sentence = self.jamo_convertor.h_2_hcj(sentence, self.vocab.pad[0])

        if sentence.__len__() < length and padding is True:
            self.__add_padding_token(sentence, length)
            # coll by object reference & mutable variable type ( immutable 할 경우 return이 꼭 포함 되어야 함. )
        return sentence

    def __add_padding_token(self, token_list: list[JamoSet], length: int):
        for i in range(length - token_list.__len__()):
            token_list.append(
                [
                    self.vocab.pad[0],
                    self.vocab.pad[0],
                    self.vocab.pad[0]
                ]
            )

    def combine(self, token_list: list[JamoSet]) -> str:
        origin_sentence = ""
        for i in range(token_list.__len__()):
            if token_list[i][0] == self.vocab.pad[0]:
                break
            origin_sentence += self.jamo_convertor.j_string_2_h(jamo_set=token_list[i], padding_token=self.vocab.pad[0],
                                                                unknown_token=self.vocab.unk[0])
        return origin_sentence

    def __zero_one_hot_vector(self, length: int):
        return [0 for _ in range(length)]

    def __one_hot_vector(self, token: str):
        one_hot_vector = self.__zero_one_hot_vector(self.vocab.length())

        if self.vocab.pad[0] == token:
            return one_hot_vector

        i = self.vocab.get_id(token)
        if is_jamo(token) or i == self.vocab.unk[1]:
            one_hot_vector[i] = np.float64(1)
        else:
            one_hot_vector[i] = np.float64(1 / 3)
        return one_hot_vector

    def encode(self, token_list: list[JamoSet]) -> list[SparseSet]:
        for i in range(len(token_list)):
            token_list[i][0] = self.__one_hot_vector(token_list[i][0])
            token_list[i][1] = self.__one_hot_vector(token_list[i][1])
            token_list[i][2] = self.__one_hot_vector(token_list[i][2])
        return list[SparseSet](token_list)

    def __get_token(self, one_hot_vector: SparseList):
        token_id: int = self.vocab.pad[1]
        for index, location_point in enumerate(one_hot_vector):
            if location_point > 0:
                token_id = index
                break
        return self.vocab.get_token(token_id)

    def decode(self, token_id_list: list[SparseSet]) -> list[JamoSet]:
        for i in range(len(token_id_list)):
            token_id_list[i][0] = self.__get_token(token_id_list[i][0])
            token_id_list[i][1] = self.__get_token(token_id_list[i][1])
            token_id_list[i][2] = self.__get_token(token_id_list[i][2])
        return list[JamoSet](token_id_list)

    def save_vocab(self, path: Union[str, os.PathLike[str]] = None) -> None:
        special_token = {}

        if path == None:
            path = "./vocab.json"

        for token, token_id in self.vocab.special_token().items():
            if token == self.vocab.pad[0]:
                special_token["pad"] = {"token_id": token_id, "using_token_name": token, "is_default": True}
            elif token == self.vocab.unk[0]:
                special_token["unk"] = {"token_id": token_id, "using_token_name": token, "is_default": True}
            else:
                special_token[token] = {"token_id": token_id, "is_default": False}

        with open(path, mode="w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "special_token": special_token,
                        "vocab": self.vocab.get_vocab()
                    },
                    ensure_ascii=False
                ))
