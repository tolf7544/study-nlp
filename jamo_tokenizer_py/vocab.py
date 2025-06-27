import inspect
from time import sleep

from util.logger import Log


class Vocab:
    __token_dict: dict[str, int]
    __id_dict: dict[int, str]
    __special_token_dict: dict[str, int]

    __log: Log
    pad: [str, int] # default token
    unk: [str, int] # default token

    def __init__(self, vocab_dict: dict[str, int] = None, special_token_list: list[int] = None,
                 padding_token: int = None, unknown_token: int = None, log: Log = Log):

        self.__log = log

        if not vocab_dict or vocab_dict.__len__() == 0:
            self.__vocab = {}
        else:
            self.__token_dict = vocab_dict
            self.__id_dict = {v: k for k, v in vocab_dict.items()}

        # pad 토큰이나 unk 토큰이 매개변수로 입력되지 않았거나 기존 vocab에 존재하지 않을 경우
        if not padding_token or not self.__id_dict.get(padding_token):
            self.__log.exit_log( "vocab.py",inspect.currentframe().f_code.co_name,"padding token must be set.")

        if not unknown_token or not self.__id_dict.get(unknown_token):
            self.__log.exit_log("vocab.py", inspect.currentframe().f_code.co_name, "unknown token must be set.")

        self.pad = [padding_token, self.__id_dict.get(padding_token)]
        self.unk = [unknown_token, self.__id_dict.get(unknown_token)]

        # 커스텀 스페셜 토큰을 __special_token_dict에 저장
        for token_id in special_token_list:
            t = self.get_token(token_id)
            if self.unk[0] == t: # 토큰이 vocab에 저장되어 있지 않을 경우
                self.__log.exit_log("vocab.py", inspect.currentframe().f_code.co_name, "special_token must be include in vocab_dict.")
            self.__special_token_dict[token_id] = t

    def get_id(self, token: str):
        t = self.__token_dict.get(token)
        if not t:
            return self.unk[1]
        else:
            return t

    def get_token(self, token_id: int):
        t = self.__id_dict.get(token_id)
        if not t:
            return self.unk[0]
        else:
            return t

    def special_token(self) -> dict[str, int]:
        return self.__special_token_dict

    def get_vocab(self) -> dict[str, int]:
        return self.__token_dict
