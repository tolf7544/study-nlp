#https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=268
# 인공지능 윤리 연구를 위한 비정형 텍스트 데이터셋
import os
import re
import sys
import unicodedata
from corpus.type import Corpus, NormalizerOption
from util import Debug


class DataNormalizers:
    method_queue: list[NormalizerOption]
    corpus: Corpus
    debug: Debug

    def __init__(self):
        self.debug = Debug(*eval(os.environ.get('DEBUG_OPTION')))
        self.method_queue = []
        self.corpus = ["sd"]  #   list[str] - default | numpy.ndarray | torch.Tensor

        self.add_all_method()  # default normalized option

    # numpy type 및 torch tensor 타입을 입력값으로 받는 로직 - _정리
    def set_corpus(self, corpus: Corpus):
        r"""말뭉치 데이터 입력"""
        self.corpus = corpus

    def get_queued_method(self):
        r"""대기중인 정규화 작업 목록 출력"""
        return self.method_queue

    def get_corpus(self):
        r"""입력된 말뭉치 데이터 출력"""
        return self.corpus

    def add_all_method(self):
        r"""모든 정규화 기능을 정규화에 적용되는 method queue에 추가"""
        for p in NormalizerOption:
            self.method_queue.append(p)

    def add_clean_text(self):
        r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
        self.method_queue.append(NormalizerOption.CLEAN_TEXT)

    # 자연어 정규화 과정에서 이메일 또는 url 형식으로 비속어를 작성 시 탐지하기 어렵기에 사전학습을 통한 자연어 이해성 높이는 방향으로 변경
    # def add_remove_url(self):
    #     r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
    #     self.method_queue.append(NormalizerOption.REMOVE_URL)

    # def add_remove_email(self):
    #     r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
    #     self.method_queue.append(NormalizerOption.REMOVE_EMAIL)

    def add_remove_repetition_char(self):
        r"""해당 메서드를 정규화를 진행하기 위해 대기 중인 method queue에 추가"""
        self.method_queue.append(NormalizerOption.REMOVE_REPETITION_CHAR)

    # def __remove_url(self, sentence:str):
    #     r"""https 또는 http로 시작하는 url 제거"""
    #     #https://stackoverflow.com/questions/9760588/how-do-you-extract-a-url-from-a-string-using-python
    #
    #     cleansing_data = re.sub(pattern=r'(https?:\/\/[^\s]+)', repl='', string=sentence)
    #     return cleansing_data

    def __remove_N_repetition_char(self, depth_count, sentence, sub_target_regex_pattern, replace_char_regex_pattern):
        r"""입력된 정규표현식을 기준으로 N개의 모든 반복문자를 탐지 및 정규화된 문자열로 치환"""
        if depth_count == sys.getrecursionlimit() - 1:
            return sentence

        target_regex = re.compile(sub_target_regex_pattern)
        replace_regex = re.compile(replace_char_regex_pattern)

        n_char_repetition = target_regex.search(sentence)

        if n_char_repetition is None:
            return sentence
        else:
            # 탐지대상이 2번 포함된 문장이 단일 선택되었을 경우 처리 로직 (ex. abcdabcdabcd에서는 선택하려는 길이의 문자가 중복으로 최소 2번 나타나야하기 때문에 abcdabcd로 나타나게 된다.
            # 이때 원래 목적인 단일 대상인 abcd로 변환하기 위해 아래와 같은 과정을 거친다.
            replace_char = replace_regex.search(sentence)
            replace_char_len = len(replace_char.group())
            if replace_char_len % 2 == 0:
                check_target = replace_char.group()[:int(replace_char_len / 2)]

                if re.findall(fr"{check_target}", replace_char.group()).__len__() == 2:
                    n_char_repetition = target_regex.sub(check_target, sentence)
            else:
                n_char_repetition = target_regex.sub(replace_char.group(), sentence)

            return self.__remove_N_repetition_char(depth_count + 1, n_char_repetition, target_regex, replace_regex)

    def __clean_text(self, sentence: str) -> str:
        r"""bert normalizer의 clean text 기능 구현"""
        #https://stackoverflow.com/questions/26741455/how-to-remove-control-characters-from-string
        regex = re.compile(r"[\u0000-\u001F\u007F-\u009F]/g")
        sentence = regex.sub(" ", sentence)
        regex = re.compile(r"[\n\r\t]")
        sentence = regex.sub(" ", sentence)

        r"""1개 이상의 연속된 공백 문자를 1개의 공백문자로 변환한다."""
        detect_whitespace_between_character_regex = re.compile(r"[ ]{2,}")
        detect_whitespace_begin_or_end_regex = re.compile(r"^( )+|( )+$")
        sentence = detect_whitespace_between_character_regex.sub(" ", sentence)
        sentence = detect_whitespace_begin_or_end_regex.sub("", sentence)

        r"""성조 제거
        유니코드 정규화 후 성조 단위로 호환 분해 후 String 형태로 출력
        """
        sentence = unicodedata.normalize('NFKD', sentence)
        sentence = ''.join([c for c in sentence if not unicodedata.combining(c)])

        return sentence

    def __normalize_UFC(self, sentence: str) -> str:
        r"""
        해당 메서드는 최종 출력 값 형태를 변경하는 역할로 queue에 적용하는 방식이 아닌 return_format에 관여한다

        :return: 정규화된 요소가 합쳐진 String
        """
        result = unicodedata.normalize("NFC", sentence)
        return result

    #파이썬 터미널에서 출력할 때 UFD로 분해된 문장은 합쳐져 출력된다.
    #만약 UFD가 진행되었는지 확인하기 위해서는 문자열의 index에 접근하여 자모단위의 출력이 발생하는 지 확인할 것
    def __normalize_UFD(self, sentence: str) -> str:
        r"""
        해당 메서드는 최종 출력 값 형태를 변경하는 역할로 queue에 적용하는 방식이 아닌 return_format에 관여한다

        :return: 유니코드 정규화 후 jamo 단위로 분해된 String
        """
        result = unicodedata.normalize("NFD", sentence)
        return result

    # def __remove_email(self, sentence: str):
    #     cleansing_data = re.sub(pattern=r'/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/.', repl='', string=sentence)
    #     return  cleansing_data

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
        thr_char_repetition = self.__remove_N_repetition_char(0, sentence, thr_char_regex_pattern,
                                                              thr_replace_char_regex_pattern)
        two_char_repetition = self.__remove_N_repetition_char(0, thr_char_repetition, two_char_regex_pattern,
                                                              two_replace_char_regex_pattern)
        one_char_repetition = self.__remove_N_repetition_char(0, two_char_repetition, one_char_regex_pattern,
                                                              one_replace_char_regex_pattern)

        return one_char_repetition

    def __run_normalize(self, method_code: list[NormalizerOption], sentence: str) -> str:
        r"""정규화 진행"""
        for code in method_code:
            if code == NormalizerOption.CLEAN_TEXT:
                sentence = self.__clean_text(sentence)
            elif code == NormalizerOption.REMOVE_REPETITION_CHAR:
                sentence = self.__remove_repetition_char(sentence)
            # elif code == NormalizerOption.REMOVE_EMAIL:
            #     sentence = self.__remove_email(sentence)
            # elif code == NormalizerOption.REMOVE_URL:
            #     sentence = self.__remove_url(sentence)

        sentence = self.__normalize_UFC(sentence)
        return sentence

    def filtering_normalize(self, sentence: str) -> str:
        r"""정규화 대기열에 입력된 정규화 함수코드를 기준으로 입력된 단일 문장 정규화 진행

        대규모 데이터 처리 시 partition으로 분활되어 처리를 할 수 있도록 지원하는 dask에 해당 메서드를 .map() 함수에
        넘겨서 사용하는 것을 추천
        """
        if isinstance(sentence, str) is False:  # 문자열이 아닌 데이터는 따로 출력하여 log파일 생성하도록 해야함
            self.debug.debug_print(f"**pass** {sentence}")
            return ""

        sentence = self.__run_normalize(self.method_queue, sentence)

        return sentence

    # 해당 방식은 partition 분활 과정을 거치지 않기에 대량의 데이터 처리에 비효율적이다.
    # pandas 또는 dask를 사용하여 대규모 데이터에 적합한 프로세스를 통해 전처리를 진행하는 것을 추천
    def compute_normalize(self) -> Corpus:
        r"""사전 입력된 말뭉치를 대상으로 정규화 대기열에 입력된 정규화 함수코드를 기준으로 정규화 진행"""

        if self.corpus.__len__() == 0:
            self.debug.debug_print(
                "[ warning ] corpus length is 0. compute_normalize() will not do anything.\n you should use .set_corpus(). ( if you want normalized not using .set_corpus(), using .filtering_normalized(sentence: str) )")
        for (i, sentence) in enumerate(self.corpus):
            if isinstance(sentence, str) is False:  # 문자열이 아닌 데이터는 따로 출력하여 log파일 생성하도록 해야함
                self.debug.debug_print(f"**pass** {sentence}")
                self.corpus[i] = ""
                continue
            self.corpus[i] = self.__run_normalize(self.method_queue, sentence)
        return self.corpus
