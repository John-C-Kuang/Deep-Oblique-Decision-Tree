from threading import *
from typing import Callable, Any


class multi_thread_tuning:

    @classmethod
    def __run_task_wrapper(cls, semaphore: Semaphore, task_function: Callable,
                           task_param: list[Any], record: list[Any]):
        semaphore.acquire()
        record.append(task_function(*task_param))
        semaphore.release()

    @classmethod
    def tune(cls, task_function: Callable, tasks_param: list[list[Any]],
             max_active_threads: int = 4):
        threads = []
        records = []
        semaphore = Semaphore(max_active_threads)
        for i in range(len(tasks_param)):
            thread = Thread(target=multi_thread_tuning.__run_task_wrapper,
                            args=[semaphore, task_function, tasks_param[i], records])
            thread.start()
            threads.append(thread)

        [thread.join() for thread in threads]
        return records


# Test

task = lambda x, y: x + y
tasks_param = [[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2]]
result = multi_thread_tuning.tune(task, tasks_param, 4)
