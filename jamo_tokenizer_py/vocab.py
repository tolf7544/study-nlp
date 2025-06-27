from time import sleep

from util.logger import Log


class Vocab:
    __token_dict: dict[str, int]
    __id_dict: dict[int, str]

    __log: Log
    pad: [str, int]
    unk: [str, int]
    mask: [str, int]

    def __init__(self, vocab_dict: dict[str, int] = None,
                 padding_token: int = None, unknown_token: int = None,
                 masking_token: int = None, log: Log = Log):

        self.__log = log

        if not vocab_dict or vocab_dict.__len__() == 0:
            self.__vocab = {}
        else:
            self.__token_dict = vocab_dict
            self.__id_dict = {v: k for k, v in vocab_dict.items()}

        self.pad = [padding_token, self.__id_dict[padding_token]]
        self.unk = [unknown_token, self.__id_dict[unknown_token]]
        self.mask = [masking_token, self.__id_dict[masking_token]]



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

    def get_vocab(self) -> dict[str, int]:
        return self.__token_dict

