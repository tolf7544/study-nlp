import json
import os
from typing import Union, Literal

import numpy as np
from jamo import is_jamo
from tokenizer.jamo_converter import JamoConvertor
from tokenizer.normalizer import Normalizer
from tokenizer.vocab import Vocab
from util.logger import Log
from util.type import JamoSet


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

    def __get_id_info(self, token: str):
        r"""
        down-scale은 알파벳, 특수문자와 같이 한글에서 하나의 문자를 이루는 자모와 같은 정보를 가지고 있으나
        하나의 문자로 모두 표현되지 않는 요소 이기에 자모와 자모외의 문자(ex. 알파벳)을 동등한 정보로 처리하기 위해
        정보를 down_scaling하는 여부를 출력요소에 포함 시킨다.
        """
        if self.vocab.pad[0] == token:
            return {
                "id": self.vocab.pad[1],
                "is_down_scaling": False,
            }

        i = self.vocab.get_id(token)
        if isinstance(token, int) == True:
            return {
                "id": i,
                "is_down_scaling": True
            }
        elif is_jamo(token) or i == self.vocab.unk[1] or self.vocab.is_special_token(token) == True:
            return {
                "id": i,
                "is_down_scaling": False
            }
        else:
            return {
                "id": i,
                "is_down_scaling": True
            }

    def encode(self, token_list: list[JamoSet], return_attention_mask: bool = False) -> object:
        result = {}
        down_scaling_mask = []
        attention_mask = []
        attention_branch_index: int = 0
        encode = []
        for i in range(len(token_list)):
            encode.append([])
            if token_list[i][0] == self.vocab.pad[0]:
                encode[i] = [self.vocab.pad[1],
                             self.vocab.pad[1],
                             self.vocab.pad[1]]
                if attention_branch_index == 0:
                    attention_branch_index = i
                down_scaling_mask.append(0)
                continue

            for j in range(3):
                id_info = self.__get_id_info(token_list[i][j])
                encode[i].append(id_info["id"])
                if j == 0:
                    if id_info["is_down_scaling"] == True:
                        down_scaling_mask.append(1)
                    else:
                        down_scaling_mask.append(0)

        result["encode"] = encode
        result["down_scaling_mask"] = down_scaling_mask
        if return_attention_mask == True:
            for i in range(len(token_list)):
                if i >= attention_branch_index:
                    attention_mask.append(0)
                else:
                    attention_mask.append(1)
            result["attention_mask"] = attention_mask
        return result


    def decode(self, token_id_list: list[list[int]]) -> list[JamoSet]:
        decode = []
        for i in range(len(token_id_list)):
            decode.append([])
            decode[i].append(self.vocab.get_token(token_id_list[i][0]))
            decode[i].append(self.vocab.get_token(token_id_list[i][1]))
            decode[i].append(self.vocab.get_token(token_id_list[i][2]))
        return list[JamoSet](decode)

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
