from log import pr_t_1, printTestLog, pr_t_2
from tokenizer.jamo.vocab import Vocab
from tokenizer.type import DefaultSpecialToken
from util import printl

from dotenv import load_dotenv

# .env를 사용할 수 있도록 함
load_dotenv()

class VocabTest():
    __vocab_sample:dict[int, str] = {0: "h", 1: "e", 2: "l", 3: "o"}
    __vocab: Vocab

    total_case_count = 0
    method_case_count = 0
    method_issue_count = 0

    def vocab_initalize_args_type(self):
        printl("__init__ 메서드 실패 케이스 테스트 시작")
        # Vocab의 arguments값을 다양한 경우의 타입으로 대입하여 정상적인 오류를 출력하는지 확인
        fail_case = [{"a": 1}, {}, 1, "a", None]
        success_case = [{1: "a"}]

        for _object in success_case:
            printl(f"성공 케이스")
            pr_t_1(f"입력: {_object}")
            pr_t_1(f"출력: {Vocab(_object)}")
            self.total_case_count += 1
            # 추후엔 코드로 통일화
        for _object in fail_case:
            printl("실패 케이스")
            pr_t_1(f"입력: {_object}")
            Vocab(_object)
            self.total_case_count += 1
            # 추후엔 코드로 통일화
        printl("vocab_initalize_args_type 종료")

    def add_object(self):

        printl("오브젝트 추가 테스트 시작")
        self.__vocab = Vocab(self.__vocab_sample)

        self.total_case_count += 1
        self.method_case_count += 1
        if self.__vocab != self.__vocab_sample:
            self.method_issue_count += 1
        print("입력 결과:", self.__vocab)
        printl("add_object 종료")

    def access_object(self):
        printl("오브젝트 접근 테스트 시작")
        print(self.__vocab)
        print("1. vocab에 값이 존재할 경우")
        res = self.__vocab.has("h")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != True:
            self.method_issue_count += 1

        printTestLog("true", res)
        print("2. vocab에 값이 없을 경우")
        res = self.__vocab.has("g")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != False:
            self.method_issue_count += 1

        printTestLog("false",res )

        printl("Method .add(token: str) 확인")
        print(self.__vocab)
        print("1. vocab에 값이 존재할 경우")
        res = self.__vocab.add("h")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != False:
            self.method_issue_count += 1
        printTestLog(False, res)

        print("2. vocab에 값이 없을 경우")
        res = self.__vocab.add("g")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != True:
            self.method_issue_count += 1
        printTestLog(True, res)
        print(self.__vocab)

        printl("Method .token_2_key(token: str) 확인")
        print(self.__vocab)
        print("1. vocab에 token: \"h\" 존재할 경우")
        res = self.__vocab.token_2_key("h")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != 0:
            self.method_issue_count += 1
        printTestLog("0", res)
        print("2. vocab에 token: \"e\"이 없을 경우")
        res = self.__vocab.token_2_key("f")
        self.total_case_count += 1
        self.method_case_count += 1
        if res != DefaultSpecialToken.UNKNOWN_TOKEN:
            self.method_issue_count += 1
        printTestLog("DefaultSpecialToken.UNKNOWN_TOKEN", res)

        printl("Method .key_2_token(key: str) 확인")
        print(self.__vocab)
        print("1. vocab에 key: 0 이 존재할 경우")
        res = self.__vocab.key_2_token(0)
        self.total_case_count += 1
        self.method_case_count += 1
        if res != "h":
            self.method_issue_count += 1
        printTestLog("h", res)
        print("2. vocab에 key: 6 없을 경우")
        res = self.__vocab.key_2_token(8)
        self.total_case_count += 1
        self.method_case_count += 1
        if res != DefaultSpecialToken.UNKNOWN_TOKEN:
            self.method_issue_count += 1
        printTestLog("DefaultSpecialToken.UNKNOWN_TOKEN", self.__vocab.key_2_token(8))

        printl("__setitem__() 확인")
        print("1. self[10] = \"f\" ( 신규 데이터 삽입 ) ")

        print("입력 결과:")
        self.__vocab[1] = "3"
        self.total_case_count += 1
        self.method_case_count += 1
        if list(self.__vocab.items())[-1][1] != "3":
            self.method_issue_count += 1
        print(self.__vocab)
        printl("access_object 종료")

    def total_count(self):
        print(f"total test count: {self.total_case_count}")
        print(f"method test count: {self.method_case_count}")
        print(f"issue occured method count: {self.method_issue_count}")
if __name__ == '__main__':
    test_case = VocabTest()

    test_case.vocab_initalize_args_type()
    test_case.add_object()
    test_case.access_object()
    test_case.total_count()