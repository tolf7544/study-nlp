import json
from typing import TypeVar, Generic

from logger.logger import Log

K = TypeVar('K')
V = TypeVar('V')

class Vocab(Generic[K,V]):
    __vocab: dict[K, V]
    __special_token: dict[V, K]
    __log: Log
    def __init__(self, row_vocab: dict[K, V], special_token: dict[V, K], log):
        self.__special_token = [] # default is zero length
        self.__vocab = [] # default is zero length
        self.__log = log

        if not isinstance(row_vocab, dict):
            log

    def set_special_token(self, special_token_set: dict[V, K]):
        if self.__special_token.__len__() == 0:
            pass