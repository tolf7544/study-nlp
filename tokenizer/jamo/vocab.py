from __future__ import annotations
import os

from tokenizer.type import DefaultSpecialToken, CustomSpecialToken, SpecialToken
from util import Debug



# 대입 연산자를 통한 연산은 불가능 하도록 정하여야 함.
class Vocab(dict):
    __reverse_dict: dict[str, int]
    special_token_list: list[SpecialToken]
    debug: Debug

    def __init__(self, _object: dict[int, str], special_token: list[CustomSpecialToken]):
        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))
        self.__reverse_dict = {}
        self.special_token = []
        self.__set_special_tokens(special_token)

        if not isinstance(_object, dict):
            self.debug.unexpected_error("Vocab.__init__(...) argument(_object) must be dict format. ")
            return

        if _object.__len__() > 0:
            test_pair = list(_object.items())[0]
            if isinstance(test_pair[0], int) and isinstance(test_pair[1], str):
                super().__init__(_object)
                self.__reverse_dict ={token: key for (key, token) in _object.items()}
            else:
                self.debug.unexpected_error("Vocab.__init__(...) argument(_object) key,value type must be {key: int, value: str}.")
        else:
            self.debug.opt_debug_print("vocab is empty.")

    def __setitem__(self, key, value):
        self.debug.debug_print("redirected add() method (prevent duplicate key or value)")
        return self.add(value)

    def __delitem__(self, key):
        self.debug.unexpected_error(
            "The vocab class is designed to allow only reading and writing to prevent unintentional errors.\n"
            "( If you need to modify or delete data from the vocab, you must access the vocab file directly. )")

    def __set_special_tokens(self, custom_special_token: list[CustomSpecialToken])-> bool:
        r"""
        vocab 인스턴스 생성 시 초기 dictionary에 special token 정보를 추가하는 메서드
        해당 메서드는 vocab의 규칙을 준수하지 않기에 vocab의 __len__()이 0일 때만 실행된다.
        """
        token_name_set = [*DefaultSpecialToken._member_names_,*custom_special_token]
        self.special_token = [name for name in token_name_set]

        if self.__len__() == 0:
            for i, token in enumerate(self.special_token):
                super()[i] = token
                self.__reverse_dict[token] = i
            return True
        else:
            return False

    def has(self, token: str) -> bool:
        r"""true: 값이 존재함. false: 값이 존재하지 않음"""
        return self.__reverse_dict.get(token) is not None

    def add(self, token: str) -> bool:
        key = self.__len__()

        if self.has(token):
            return False

        super().__setitem__(key, token)
        self.__reverse_dict[token] = key
        return  True

    def token_2_key(self, token: str):
        val = self.__reverse_dict.get(token)
        if val is None:
            return DefaultSpecialToken.UNK
        else:
            return val

    def key_2_token(self, key: int):
        val = self.get(key)
        if val is None:
            return DefaultSpecialToken.UNK
        else:
            return val