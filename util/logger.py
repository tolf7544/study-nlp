import os
import time
import warnings
from datetime import datetime, timezone
from typing import Union

from typing_extensions import Optional, Literal


class Log:
    domain: str
    save_path: Optional[os.PathLike[str]]
    mode: Literal["debug", "common"]
    is_display: bool
    is_save: bool

    def __init__(self, domain_name: str = "", save_path: Optional[Union[os.PathLike[str], str]] = "./log",
                 mode: Literal["debug", "common"] = "debug", is_display: bool = True, is_save: bool = True):
        self.domain = domain_name
        self.save_path = save_path
        self.mode = mode
        self.is_display = is_display
        self.is_save = is_save

    def display(self, file_name: str, function_name: str, reason: str, data: str) -> None:
        if self.is_display and self.mode == "debug":
            print(f"[ {self.domain}.{file_name}.{function_name} ] ", end="")
            print(f"> {reason}\n> {data}")

    def save(self, file_name: str, function_name: str, reason: str, data: str, is_exit: bool = False) -> None:
        if self.is_save:
            if self.save_path == "./log" and os.path.exists("./log") == False:
                os.mkdir("./log")
            elif os.path.exists(self.save_path) == False:
                exit("log save path is not exist. check argument.save_path is exist.  ")
            else:
                pass

            if not is_exit:
                is_exit = 0
            else:
                is_exit = 1

            timestamp = time.time()
            local_time = time.localtime(timestamp)
            formatted_time = time.strftime("%Y_%m_%d-%H_%M_%S", local_time)
            increase_number = len(os.listdir(self.save_path)) + 1

            with open(
                    f"{self.save_path}/{is_exit}_{formatted_time}-{increase_number}-{self.domain}.{file_name}.{function_name}",
                    mode="w", encoding="utf-8") as f:
                f.write(f"[ {self.domain}.{file_name}.{function_name} ] reason:{reason}, data: {data}")

    def exit_log(self, file_name: str, function_name: str, reason: str) -> None:
        self.save(file_name, function_name, reason, "exit_log does not provide data.", True)
        exit(f"[ {self.domain}.{file_name}.{function_name}.exit ] {reason}")

    def warn_display(self, file_name: str, function_name: str, reason: str) -> None:
        warnings.warn(f"[ {file_name}.{function_name}.warn ] {reason} (this is not error)",
                      category=Warning, stacklevel=3)
