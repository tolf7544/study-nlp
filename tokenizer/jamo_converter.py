from typing_extensions import Union, Optional

from util.type import JamoSet
from jamo import j2hcj, h2j, j2h


class JamoConvertor:
    def __init__(self):
        pass

    def h_2_hcj(self, content: str, pad: str) -> list[JamoSet]:
        r"""
        한글 문장 > h2j(자소 단위 정규화) > j2hcj(호환 자모 정규화)
        """
        result: list[JamoSet] = []

        for char in content:
            char: list[str] = list(j2hcj(h2j(char)))
            if char.__len__() < 3:
                for i in range(3 - char.__len__()):
                    char.append(pad)
            result.append(JamoSet(char))
        return result

    def j_string_2_h(self, jamo_set: Union[Optional[str], JamoSet], padding_token: str, unknown_token: str) -> str:
        r"""(3, vocab_length) shape 형태 백터 한글 변형"""
        is_crashed: bool = False

        if jamo_set == padding_token:
            return ""

        for i in range(3):
            if jamo_set[i] == padding_token:
                jamo_set[i] = ""
            if jamo_set[i] == unknown_token:
                is_crashed = True
                jamo_set[i] = "__"

        if jamo_set[0].__len__() == 1 and jamo_set[1].__len__() == 1 and jamo_set[
            2].__len__() < 2:  # 초성 중성이 존재할 시에만 j2h 메서드 사용
            print(jamo_set)
            return j2h(*jamo_set)
        else:
            if is_crashed == True:
                return f"<{jamo_set[0]}{jamo_set[1]}{jamo_set[2]}>"
            else:
                return jamo_set[0]
