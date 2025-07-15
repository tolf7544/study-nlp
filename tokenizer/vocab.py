import inspect
from time import sleep

from jinja2.nodes import Literal

from util.logger import Log


class Vocab:
    __token_dict: dict[str, int]
    __id_dict: dict[int, str]
    __special_token_dict: dict[str, int]

    __log: Log
    pad: (str, int) # default token
    unk: (str, int) # default token

    def __init__(self, vocab_dict: dict[str, int] = None, special_token_list: list[str] = None,
                 padding_token: str = None, unknown_token: str = None, log: Log = None):

        self.__log = log

        if vocab_dict == None or vocab_dict.__len__() == 0:
            self.__token_dict = {}
            self.__id_dict = {}
            self.__special_token_dict = {}
        else:
            self.__token_dict = vocab_dict
            self.__id_dict = {v: k for k, v in vocab_dict.items()}
            self.__special_token_dict = {}
        # pad 토큰이나 unk 토큰이 매개변수로 입력되지 않았거나 기존 vocab에 존재하지 않을 경우
        if padding_token == None  or self.__token_dict.get(padding_token) == None:
            self.__log.exit_log( "vocab.py", inspect.currentframe().f_code.co_name,"padding token must be set.")

        if unknown_token == None or self.__token_dict.get(unknown_token) == None:
            self.__log.exit_log("vocab.py", inspect.currentframe().f_code.co_name, "unknown token must be set.")

        self.pad = (padding_token, self.__token_dict.get(padding_token))
        self.unk = (unknown_token, self.__token_dict.get(unknown_token))
        self.__special_token_dict[self.pad[0]] = self.pad[1]
        self.__special_token_dict[self.unk[0]] = self.unk[1]

        # 커스텀 스페셜 토큰을 __special_token_dict에 저장
        if special_token_list == None:
            return

        for token in special_token_list:
            token_id = self.get_id(token)

            if self.unk[0] != token and self.unk[1] == token_id: # 토큰이 vocab에 저장되어 있지 않을 경우
                self.__log.exit_log("vocab.py", inspect.currentframe().f_code.co_name, "special_token must be include in vocab_dict.")
            self.__special_token_dict[token] = token_id

    def get_id(self, token: str) -> int:
        t = self.__token_dict.get(token)
        if t == None:
            return self.unk[1]
        else:
            return t

    def get_token(self, token_id: int):
        t = self.__id_dict.get(token_id)
        if t == None:
            return self.unk[0]
        else:
            return t

    def is_special_token(self, token: str) -> bool:
        is_none = self.__special_token_dict.get(token)
        if is_none == None:
            return False
        else:
            return True

    def special_token(self) -> dict[str, int]:
        return self.__special_token_dict


    def get_vocab(self) -> dict[str, int]:
        return self.__token_dict

    def length(self):
        return self.__token_dict.__len__()