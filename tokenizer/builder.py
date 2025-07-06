import inspect
from typing import Any, Union

from jamo import j2hcj, h2j

from tokenizer.jamo_tokenizer import JamoTokenizer
from tokenizer.normalizer import Normalizer
from tokenizer.vocab import Vocab
from util.logger import Log
from util.type import NormalizerMethod


class TokenizerBuilder:
    padding_token: str
    unknown_token: str
    special_token: list[str]
    corpus: list[str]
    normalizing_queue: list[NormalizerMethod]
    normalizer: Normalizer

    log = Log(domain_name="VocabBuilder", save_path=".log", mode="debug", is_save=False, is_display=True)

    def __init__(self):
        self.special_token = []
        self.corpus = []

        self.normalizing_queue = []
        self.normalizer = Normalizer()

    def set_special_token(self, special_token: list[str] = None, padding_token: str = "<pad>",
                          unknown_token: str = "<unk>"):  # return VocabBuilder

        self.special_token.append(padding_token)
        self.special_token.append(unknown_token)
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        if special_token == None or special_token.__len__() == 0:
            return self

        for token in special_token:
            if isinstance(token, str):
                self.special_token.append(token)
            else:
                self.log.exit_log("builder.py", inspect.currentframe().f_code.co_name,
                                  "all special_token list element must be str")
        return self

    def set_corpus(self, corpus: Union[list[str], Any]):  # return VocabBuilder
        self.corpus = corpus
        return self

    def __normalize(self, sentence: str):
        for query in self.normalizing_queue:
            if query == NormalizerMethod.REMOVE_CONTROL_CHAR.name:
                self.normalizer.remove_control_char(sentence)

            elif query == NormalizerMethod.REDUCE_WHITESPACE.name:
                self.normalizer.reduce_whitespace(sentence)

            elif query == NormalizerMethod.REMOVE_ACCENT.name:
                self.normalizer.remove_accent(sentence)

            elif query == NormalizerMethod.REMOVE_REPETITION_CHAR.name:
                self.normalizer.reduce_all_repetition_char(sentence)

            elif query == NormalizerMethod.NFK.name:
                self.normalizer.normalization_from_combination(sentence)

            else:
                self.log.exit_log("builder.py", inspect.currentframe().f_code.co_name,
                                  "normalize literal list is including undefined string literal.")
        return sentence

    def set_normalizing(self, normalize: list[NormalizerMethod] = None):  # return VocabBuilder
        r"""
        :param normalize: 진행할 정규화 단계 추가 ( 미 입력시 모두 적용 )
        :return: VocabBuilder

        """
        #       remove_control_char: 아스키 코드 제어 문자 제거.
        #       reduce_whitespace: 띄어쓰기 N개를 1개로 정규화.
        #       remove_accent: 중국어와 같이 문자에 포함된 발음 기호 제거.
        #       reduce_repetition_char: 반복 문자 정규화
        #           반복 문자 길이(1) -> 정규화 반복 횟수: 3
        #           반복 문자 길이(2 또는 3) -> 정규화 반복 횟수: 2
        #           반복 문자 길이(4이상) -> 정규화 반복 횟수: 1
        #       NFK: 유니코드 정규화.
        if normalize == None:
            normalize = NormalizerMethod._member_names_
        self.normalizing_queue = normalize
        self.__normalize(sentence="test_sentence")  # normalizer test, 문제 발생 시 error 제공
        return self

    def build_tokenizer(self) -> Vocab:
        vocab_dict = {}

        for token in self.special_token:
            vocab_dict[token] = vocab_dict.__len__()

        for sentence in self.corpus:
            sentence = self.__normalize(sentence)
            keyword_set = list(set(j2hcj(h2j(sentence))))

            for i in range(keyword_set.__len__()):
                id = vocab_dict.get(keyword_set[i])
                if id == None:
                    vocab_dict[keyword_set[i]] = vocab_dict.__len__()
                else:
                    continue

        vocab = Vocab(vocab_dict=vocab_dict, special_token_list=self.special_token, padding_token=self.padding_token,
                      unknown_token=self.unknown_token, log=self.log)
        tokenizer = JamoTokenizer(vocab)
        tokenizer.save_vocab()
        return tokenizer
