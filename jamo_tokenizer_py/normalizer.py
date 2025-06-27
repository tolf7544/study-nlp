import re
import sys
import unicodedata


class Normalizer:
    def __replace_repetition_char(self, depth_count: int = 1, sentence: str = "", existing_regex = None, replacing_regex = None) -> str:
        r"""입력된 정규표현식을 기준으로 N개의 모든 반복문자를 탐지 및 정규화된 문자열로 치환"""
        if depth_count == sys.getrecursionlimit() - 1:
            return sentence

        target_regex = re.compile(existing_regex)
        replace_regex = re.compile(replacing_regex)

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

            return self.__replace_repetition_char(depth_count + 1, n_char_repetition, target_regex, replace_regex)

    def reduce_all_repetition_char(self, sentence: str) -> str:
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
        thr_char_repetition:str = self.__replace_repetition_char(0, sentence, thr_char_regex_pattern,
                                                              thr_replace_char_regex_pattern)
        two_char_repetition:str = self.__replace_repetition_char(0, thr_char_repetition, two_char_regex_pattern,
                                                              two_replace_char_regex_pattern)
        one_char_repetition:str = self.__replace_repetition_char(0, two_char_repetition, one_char_regex_pattern,
                                                              one_replace_char_regex_pattern)

        return one_char_repetition

    def remove_control_char(self, sentence: str):
        #https://stackoverflow.com/questions/26741455/how-to-remove-control-characters-from-string
        regex = re.compile(r"[\u0000-\u001F\u007F-\u009F]/g")
        sentence = regex.sub(" ", sentence)
        regex = re.compile(r"[\n\r\t]")
        sentence = regex.sub(" ", sentence)
        return sentence

    def reduce_whitespace(self, sentence: str):
        r"""1개 이상의 연속된 공백 문자를 1개의 공백문자로 변환한다."""
        detect_whitespace_between_character_regex = re.compile(r"[ ]{2,}")
        detect_whitespace_begin_or_end_regex = re.compile(r"^( )+|( )+$")
        sentence = detect_whitespace_between_character_regex.sub(" ", sentence)
        sentence = detect_whitespace_begin_or_end_regex.sub("", sentence)
        return sentence


    def remove_accent(self, sentence: str):
        r"""성조 제거
        유니코드 정규화 후 성조 단위로 호환 분해 후 String 형태로 출력
        """
        return ''.join([c for c in sentence if not unicodedata.combining(c)])

    def nomalization_from_combination(self, sentence: str):
        return unicodedata.normalize('NFKD', sentence)
