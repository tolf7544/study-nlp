#https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=268
# 인공지능 윤리 연구를 위한 비정형 텍스트 데이터셋
import re
from typing import Union, Optional

import numpy
import torch
from sympy.strategies.core import switch

from corpus.type import TypeCorpus, NormalizationMethod


class DataNormalizers():
    __method_queue:list[NormalizationMethod]
    __corpus: TypeCorpus
    def __init__(self):
        self.__method_queue = []
        self. __corpus = []
    # numpy type 및 torch tensor 타입을 입력값으로 받는 로직 - _정리
    def set_corpus(self, corpus: TypeCorpus):
        r"""말뭉치 데이터 입력"""
        self.__corpus = corpus

    def get_queued_method(self):
        r"""대기중인 정규화 작업 목록 출력"""
        return self.__method_queue

    def get_corpus(self):
        r"""입력된 말뭉치 데이터 출력"""
        return self.__corpus

    def add_all_method(self):
        r"""모든 정규화 기능을 정규화에 적용되는 method queue에 추가"""
        for property in NormalizationMethod:
            self.__method_queue.append(property)

    def add_remove_linebreaks(self):
        r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
        self.__method_queue.append(NormalizationMethod.REMOVE_LINEBREAKS)

    def add_remove_whitespace(self):
        r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
        self.__method_queue.append(NormalizationMethod.REMOVE_WHITESPACE)

    # def add_remove_url(self):
    #     r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
    #     self.__method_queue.append(NormalizationMethod.REMOVE_URL)

    # def add_remove_email(self):
    #     r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
    #     self.__method_queue.append(NormalizationMethod.REMOVE_EMAIL)

    def add_remove_repetition_char(self):
        r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
        self.__method_queue.append(NormalizationMethod.REMOVE_REPETITION_CHAR)

    def __remove_url(self, sentence:str):
        r"""https 또는 http로 시작하는 url 제거"""
        #https://stackoverflow.com/questions/9760588/how-do-you-extract-a-url-from-a-string-using-python

        cleansing_data = re.sub(pattern=r'(https?:\/\/[^\s]+)', repl='', string=sentence)
        return cleansing_data

    def __remove_N_repetition_char(self, depth_count, sentence, sub_target_regex_pattern, replace_char_regex_pattern):
        r"""입력된 정규표현식을 기준으로 N개의 모든 반복문자를 탐지 및 정규화된 문자열로 치환"""
        if depth_count == 30:
            return "overflow"

        target_regex = re.compile(sub_target_regex_pattern)
        replace_regex = re.compile(replace_char_regex_pattern)

        N_char_repetiton = target_regex.search(sentence)

        if N_char_repetiton == None:
            return sentence
        else:
            # 탐지대상이 2번 포함된 문장이 단일 선택되었을 경우 처리 로직 (ex. abcdabcdabcd에서는 선택하려는 길이의 문자가 중복으로 최소 2번 나타나야하기 때문에 abcdabcd로 나타나게 된다.
            # 이때 원래 목적인 단일 대상인 abcd로 변환하기 위해 아래와 같은 과정을 거친다.
            replace_char = replace_regex.search(sentence)
            replace_char_len = len(replace_char.group())
            if replace_char_len % 2 == 0:
                check_target = replace_char.group()[:int(replace_char_len / 2)]

                if re.findall(fr"{check_target}", replace_char.group()).__len__() == 2:
                    N_char_repetiton = target_regex.sub(check_target, sentence)
            else:
                N_char_repetiton = target_regex.sub(replace_char.group(), sentence)

            return self.__remove_N_repetition_char(depth_count + 1, N_char_repetiton, target_regex, replace_regex)

    def __remove_linebreaks(self, sentence: str):
        r"""\n 또는 \r와 같이 문단을 나누는 요소를 공백문자 한개로 변환한다."""
        detect_linebreaks_regex = re.compile(r"\r\n|\n|\r")
        sentence = detect_linebreaks_regex.sub(" ",sentence)
        return sentence

    def __remove_whitespace(self, sentence: str):
        r"""1개 이상의 연속된 공백 문자를 1개의 공백문자로 변환한다."""
        detect_whitespace_between_character_regex = re.compile(r"[ ]{2,}")
        detect_whitespace_begin_or_end_regex = re.compile(r"^( )+|( )+$")
        sentence = detect_whitespace_between_character_regex.sub(" ",sentence)
        sentence = detect_whitespace_begin_or_end_regex.sub("",sentence)
        return sentence

    def __remove_repetition_char(self, sentence: str):
        r"""모든 반복문자를 탐지 및 정규화된 문자열로 치환"""
        #https://f7project.tistory.com/383
        # 아래 두 정규표현식 모두 위의 블로그를 참조함
        #https://stackoverflow.com/questions/17680631/python-regular-expression-not-matching
        one_char_regex_pattern = r'(\S{1}?)\1{3,}'
        two_char_regex_pattern = r'(\S{2,3}?)\1{2,}'
        thr_char_regex_pattern = r'(\S{4}?)\1{1,}'
        one_replace_char_regex_pattern = r'(\S{1}?)\1{2}'
        two_replace_char_regex_pattern = r'(\S{2,3}?)\1'
        thr_replace_char_regex_pattern = r'(\S{4}?)\1'

        # 반복문자 1개 탐지는 반복문자 2개 탐지의 부분 집합이다.
        # 또한 반복문자 2개 집합은 반복문자 3개 이상의 집합과 교집합이 발생한다. (부분 집합 x)

        # 그렇기에 반복문자 1개를 처리하는 것은 마지막 순서가 되고
        # 그 다음으로 2개, 3개 이상을 위치해야 필요없는 연산을 막을 수 있다.
        thr_char_repetition = self.__remove_N_repetition_char(0, sentence, thr_char_regex_pattern, thr_replace_char_regex_pattern)
        two_char_repetition = self.__remove_N_repetition_char(0, thr_char_repetition, two_char_regex_pattern, two_replace_char_regex_pattern)
        one_char_repetition = self.__remove_N_repetition_char(0, two_char_repetition, one_char_regex_pattern, one_replace_char_regex_pattern)

        return one_char_repetition

    def __run_normalize(self, method_code: NormalizationMethod, sentence: str):
        r"""정규화 진행"""
        if method_code == NormalizationMethod.REMOVE_URL:
            sentence = self.__remove_url(sentence)
        elif method_code == NormalizationMethod.REMOVE_WHITESPACE:
            sentence = self.__remove_whitespace(sentence)
        elif method_code == NormalizationMethod.REMOVE_REPETITION_CHAR:
            sentence = self.__remove_repetition_char(sentence)
        elif method_code == NormalizationMethod.REMOVE_LINEBREAKS:
            sentence = self.__remove_linebreaks(sentence)
        else:
            ...
        return sentence

    def filtering_normalize(self, sentence: str):
        r"""정규화 대기열에 입력된 정규화 함수코드를 기준으로 입력된 단일 문장 정규화 진행"""
        if isinstance(sentence, str) is False: # 문자열이 아닌 데이터는 따로 출력하여 log파일 생성하도록 해야함
            print(sentence)
            return ""

        for method_code in self.__method_queue:
            sentence = self.__run_normalize(method_code, sentence)
        return sentence

    def compute_normalize(self):
        r"""사전 입력된 말뭉치를 대상으로 정규화 대기열에 입력된 정규화 함수코드를 기준으로 정규화 진행"""
        for method_code in self.__method_queue:
            for (i, sentence) in enumerate(self.__corpus):
                if isinstance(sentence, str) is False:  # 문자열이 아닌 데이터는 따로 출력하여 log파일 생성하도록 해야함
                    print(sentence)
                    self.__corpus[i] = ""
                    continue
                self.__corpus[i] = self.__run_normalize(method_code, sentence)
        return self.__corpus