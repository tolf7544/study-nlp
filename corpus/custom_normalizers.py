# https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=268
# 인공지능 윤리 연구를 위한 비정형 텍스트 데이터셋
from __future__ import annotations

import re
from os import error, PathLike
from tkinter.messagebox import RETRY
from typing import Union, Optional

from corpus.type import CorpusNormalizerStatus, ErrorType, NormalizerMethodResult


class SentenceNormalizers():
    __failed_normalization_contents: list[str]

    sentence_count: int


    def __init__(self):
        self.__failed_normalization_contents = list()
        self.sentence_count = 0

    def __issue_collector(self, result: NormalizerMethodResult):
        if result.status[0] == CorpusNormalizerStatus.SUCCESS:
            return result.contents
        else:
            if result.issue[0] in [ErrorType.OVERFLOW,ErrorType.DATA_ISSUE]:
                self.__failed_normalization_contents.append(result.contents)
                return ""
            else:
                exit(f"예상치 못한 활동이 발생하였습니다. 최종 정규화 함수의 결과는 다음과 같습니다.\n{result.__repr__()}")

    def __remove_url(self, sentence: str):
        # https://stackoverflow.com/questions/9760588/how-do-you-extract-a-url-from-a-string-using-python
        cleansing_data = re.sub(pattern=r'(https?:\/\/[^\s]+)', repl='', string=sentence)
        return cleansing_data

    def __remove_n_repetition_char(self, depth_count, sentence, sub_target_regex_pattern, replace_char_regex_pattern) -> NormalizerMethodResult:
        if depth_count == 30:
            return NormalizerMethodResult(_status=CorpusNormalizerStatus.FAILED_REPETITION_FILTERING,
                                          _issue=ErrorType.OVERFLOW,
                                          _contents=sentence)
        target_regex = re.compile(sub_target_regex_pattern)
        replace_regex = re.compile(replace_char_regex_pattern)

        n_char_repetition = target_regex.search(sentence)

        if n_char_repetition == None:
            return NormalizerMethodResult(_status=CorpusNormalizerStatus.SUCCESS,
                                          _contents=sentence)
        else:
            # 탐지대상이 2번 포함된 문장이 단일 선택되었을 경우 처리 로직 (ex. abcdabcdabcd에서는 선택하려는 길이의 문자가 중복으로 최소 2번 나타나야하기 때문에 abcdabcd로 나타나게 된다.
            # 이때 원래 목적인 단일 대상인 abcd로 변환하기 위해 아래와 같은 과정을 거친다.

            replace_char = replace_regex.search(n_char_repetition.group())

            replace_char_len = len(replace_char.group())
            # 2~3 범위까지는 정규표현식 범위에서 처리 가능하다. 그러기에 아래의 과정을 거치는 것을 생략한다.
            #4글자 이상부터는 해당 과정을 거쳐야 한다.
            if replace_char_len % 2 == 0 and replace_char_len > 4:
                check_target = replace_char.group()[:int(replace_char_len / 2)]

                if re.findall(check_target, replace_char.group()).__len__() == 2:
                    n_char_repetition = target_regex.sub(check_target, sentence)
            else:
                n_char_repetition = sentence.replace(n_char_repetition.group(), replace_char.group())
            return self.__remove_n_repetition_char(depth_count + 1, n_char_repetition, target_regex, replace_regex)

    def __remove_repetition_char(self, sentence: str):
        # https://f7project.tistory.com/383
        # 아래 두 정규표현식 모두 위의 블로그를 참조함
        # https://stackoverflow.com/questions/17680631/python-regular-expression-not-matching
        one_char_regex_pattern = r'(\S{1})\1{3,}'
        two_char_regex_pattern = r'(\S{2})\1{2,}'
        thr_char_regex_pattern = r'(\S{3,})\1{1,}'

        one_replace_char_regex_pattern = r'(\S{1})\1{2}'
        two_replace_char_regex_pattern = r'(\S{2})\1'
        thr_replace_char_regex_pattern = r'(\S{3,})\1'


        two_char_repetition = self.__remove_n_repetition_char(0, sentence, two_char_regex_pattern,
                                                              two_replace_char_regex_pattern)
        two_char_repetition = self.__issue_collector(two_char_repetition)
        thr_char_repetition = self.__remove_n_repetition_char(0, two_char_repetition, thr_char_regex_pattern,
                                                              thr_replace_char_regex_pattern)
        thr_char_repetition = self.__issue_collector(thr_char_repetition)
        one_char_repetition = self.__remove_n_repetition_char(0, thr_char_repetition, one_char_regex_pattern,
                                                              one_replace_char_regex_pattern)
        one_char_repetition = self.__issue_collector(one_char_repetition)
        return one_char_repetition

    def __remove_whitespace(self, sentence: str):
        detect_whitespace_between_character_regex = re.compile(r"[ ]{2,}")
        detect_whitespace_begin_or_end_regex = re.compile(r"^( )+|( )+$")
        sentence = detect_whitespace_between_character_regex.sub(" ",sentence)
        sentence = detect_whitespace_begin_or_end_regex.sub("",sentence)
        return sentence

    def get_error_contents(self, save_file: Optional[PathLike] = None):
        #파일 저장 로직 구성해야함
        return self.__failed_normalization_contents

    def normalizer(self,
                   sentences: Optional[Union[str, list[str]]],
                   disable_remove_URL: Optional[bool] = False,
                   disable_remove_repetitionChar: Optional[bool] = False,
                   disable_remove_whitespace: Optional[bool] = False) -> NormalizerMethodResult:


        if type(sentences) == str:
            sentences = [sentences]

        for i, sentence in enumerate(sentences):
            if not disable_remove_URL:
                sentence = self.__remove_url(sentence)
            if not disable_remove_repetitionChar:
                sentence = self.__remove_repetition_char(sentence)
            if not disable_remove_whitespace:
                sentence = self.__remove_whitespace(sentence)

            if sentence.__len__() > 0:
                sentences[i] = sentence
            else:
                sentences.pop(i)
        if self.__failed_normalization_contents.__len__() > 0:
            return NormalizerMethodResult(_status=CorpusNormalizerStatus.SUCCESS_BUT_ISSUE_OCCURRED,
                                          _issue=ErrorType.OVERFLOW,
                                          _contents=sentences)
        else:
            return NormalizerMethodResult(_status=CorpusNormalizerStatus.SUCCESS,
                                          _issue=ErrorType.OVERFLOW,
                                          _contents=sentences)