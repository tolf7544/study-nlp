import inspect
import json
from curses.ascii import isdigit
from typing import TypeVar, Generic

from numpy.f2py.auxfuncs import isstring
from pandas.core.computation.ops import isnumeric

from logger.logger import Log

K = TypeVar('K')
V = TypeVar('V')


class Vocab(Generic[K, V]):
    __vocab: dict[K, V]
    __special_token: dict[V, K]
    __log: Log

    def __init__(self, row_vocab: dict[K, V], special_token: dict[V, K], log):
        self.__log = log

        if row_vocab.__len__() == 0:
            self.__vocab = {}
        else:
            self.__vocab = row_vocab


    def set_special_token(self, special_token_set: dict[V, K]):
        if self.__special_token.__len__() == 0:
            self.__log.warn_display(file_name="vocab.py", function_name=inspect.currentframe().f_code.co_name,
                                    reason="special_token_set dictionary is empty.")
            return None



    def get(self, key: K):
        ...

    def add(self, key: K, value: V) -> bool:
        ...

    def get_vocab(self) -> dict[K, V]:
        ...

    def get_special_token(self) -> dict[V, K]:
        ...