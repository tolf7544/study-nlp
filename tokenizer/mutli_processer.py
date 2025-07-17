from collections.abc import Callable
from copy import deepcopy
from multiprocessing import Process, Event
from multiprocessing.queues import Queue


## 나중에 개발 ( 개발 중지 )

class MultiProcessor:
    def __init__(self, dataset: list, single_process_maximum_bytes: int = 2 ** 20):
        self.dataset = dataset
        self.single_process_maximum_bytes = single_process_maximum_bytes  # default 1mb
        self.num_process = self.__get_process_number()
        self.chunk_size = self.dataset.__len__() // self.num_process
        self.queue = Queue()
        self.result_arr = []

    def __get_process_number(self):
        if self.dataset.__sizeof__() < self.single_process_maximum_bytes:
            return 1
        elif self.dataset.__sizeof__() % self.single_process_maximum_bytes > 0:
            return self.dataset.__sizeof__() // self.single_process_maximum_bytes + 1  # 1mb를 기준으로 병렬 처리 진행.
        else:
            return self.dataset.__sizeof__() // self.single_process_maximum_bytes

    def __target_function(self, function: Callable, *args):
        result = function(*args)
        self.queue.put(result)
        args[0].set()
        exit()

    def __event_listener(self, event: Event):
        event.wait()
        self.result_arr.append(self.queue.get())


    def generate_process(self, function: Callable):
        processes = {}
        process_event = Event()

        for i in range(self.num_process):
            __maximum_index = (i + 1) * self.chunk_size
            __minimum_index = i * self.chunk_size

            process = Process(target=deepcopy(self.__target_function),
                        args=(process_event, self.dataset[__minimum_index:__maximum_index]))
            processes[process.pid] = process




