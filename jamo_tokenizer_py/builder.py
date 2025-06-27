from typing import Literal

from vocab import Vocab


class VocabBuilder:
    special_token: list[str]

    def set_special_token(self, special_token: list[str]):  # return VocabBuilder
        ...
        return self

    def set_corpus(self, corpus: list[str]):  # return VocabBuilder
        ...
        return self

    def set_normalizing(self, normalize: Literal[""]):  # return VocabBuilder
        ...
        return self

    def build_vocab(self) -> Vocab:
        ...
