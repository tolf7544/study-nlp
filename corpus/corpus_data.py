#https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=268
# 인공지능 윤리 연구를 위한 비정형 텍스트 데이터셋
import re


class Corpus():
    def __init__(self):
        ...
    def remove_url(self, content:str):
        #https://stackoverflow.com/questions/9760588/how-do-you-extract-a-url-from-a-string-using-python
        cleansing_data = re.sub(pattern=r'(https?:\/\/[^\s]+)', repl='', string=content)
        return cleansing_data

    def remove_N_repetition_char(self, depth_count, sentence, sub_target_regex_pattern, replace_char_regex_pattern):
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

            return self.remove_N_repetition_char(depth_count+1, N_char_repetiton, target_regex, replace_regex)

    def remove_repetition_char(self, sentence: str):
        #https://f7project.tistory.com/383
        # 아래 두 정규표현식 모두 위의 블로그를 참조함
        #https://stackoverflow.com/questions/17680631/python-regular-expression-not-matching
        one_char_regex_pattern = r'(\S{1}?)\1{3,}'
        two_char_regex_pattern = r'(\S{2,3}?)\1{2,}'
        thr_char_regex_pattern = r'(\S{4}?)\1{1,}'
        one_replace_char_regex_pattern = r'(\S{1}?)\1{2}'
        two_replace_char_regex_pattern = r'(\S{2,3}?)\1'
        thr_replace_char_regex_pattern = r'(\S{4}?)\1'

        thr_char_repetition = self.remove_N_repetition_char(0, sentence, thr_char_regex_pattern, thr_replace_char_regex_pattern)
        two_char_repetition = self.remove_N_repetition_char(0, thr_char_repetition, two_char_regex_pattern, two_replace_char_regex_pattern)
        one_char_repetition = self.remove_N_repetition_char(0, two_char_repetition, one_char_regex_pattern, one_replace_char_regex_pattern)


        # for i in range():
        #     # 하나의 문자가 반복될 때 처리
        #
        #     cleansing_data = re.sub(pattern=r'(\S{1}?)\1{3,}', repl=r'(\S{1}?)\1{3}', string=content)
        #
        #     # # 문자 갯수가 2 이상인 반복 문자열 처리
        #     # cleansing_data = re.sub(pattern=r'(\S{2,3}?)\1{2,}', repl=r'(\S{2,3}?)\1{1}', string=cleansing_data)
        #     #
        #     # # 문자 갯수가 4 이상인 반복 문자열 처리
        #     # cleansing_data = re.sub(pattern=r'(\S{4,}?)\1', repl=r'(\S{4}?)', string=cleansing_data)
        return one_char_repetition