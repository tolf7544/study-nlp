def printl(sentence: str):
    r"""print log(<str>) 입력값 상단 ( =*20 )을 위치시킵니다."""
    print(f"********************************\n{sentence}")

class Debug():
    is_optional_log_display: bool
    is_log_display: bool
    is_test_mode: bool
    def __init__(self,
                 is_optional_log_display: bool = False, is_log_display: bool = True, is_test_mode: bool = False):
        r"""
        :param is_optional_log_display: true일 시 opt_debug_print를 콘솔에 출력합니다.
        :param is_log_display: true일 시 debug_print를 콘솔에 출력합니다.
        """

        self.is_log_display = is_log_display
        self.is_optional_log_display = is_optional_log_display
        self.is_test_mode = is_optional_log_display
    def opt_debug_print(self,msg):
        if self.is_optional_log_display == True:
            print(f"[log] {msg}")

    def debug_print(self, msg):
        if self.is_log_display == True:
            print(f"{msg}")

    def unexpected_error(self, msg: str):
        if self.is_test_mode == True:
            print(msg)
        else:
            exit(msg)

    # def create_analysis_object(self):
    #     ...
        # 성공 / 실패 분석 및 통계를 위한 오브젝트 제공 디버깅 함수