import os.path
from typing import Literal

from logger.logger import Log

log_debug = Log(domain_name="test", mode="debug", is_display=True, is_save=True)
log_common = Log(domain_name="test", mode="common", is_display=True, is_save=True)

log_debug_display_false = Log(domain_name="test", mode="debug", is_display=False, is_save=True)
log_common_display_false = Log(domain_name="test", mode="common", is_display=False, is_save=True)

log_debug_is_save_false = Log(domain_name="test", mode="debug", is_display=True, is_save=False)
log_common_is_save_false = Log(domain_name="test", mode="common", is_display=True, is_save=False)

def test_display():
    print("일반 상태")
    log_debug.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터1")
    log_common.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터2")
    print("\n")
    print("display_false 상태")
    log_debug_display_false.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터3")
    log_debug_display_false.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터4")
    print("\n")
    print("is_save_false 상태")
    log_debug_is_save_false.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터5")
    log_debug_is_save_false.display(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터6")
    print("\n")

def test_save():
    print("일반 상태")
    log_debug.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터1")
    log_common.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터2")
    print("\n")
    print("display_false 상태")
    log_debug_display_false.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터3")
    log_debug_display_false.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터4")
    print("\n")
    print("is_save_false 상태")
    log_debug_is_save_false.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터5")
    log_debug_is_save_false.save(file_name="logger", function_name="test_func", reason="테스트", data="테스트 데이터6")
    print("\n")
    
if __name__ == '__main__':
    #test_display() 통과
    test_save()