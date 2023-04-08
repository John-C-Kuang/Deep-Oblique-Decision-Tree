import os
import signal
from multiprocessing import Process, Manager, BoundedSemaphore
from typing import Callable, Any, Sequence


class multi_process_tuning:

    @classmethod
    def _run_task_wrapper(cls, semaphore: BoundedSemaphore, task_function: Callable,
                           task_param: list[Any], records: list[Any]):
        semaphore.acquire()
        records.append(task_function(*task_param))
        semaphore.release()
        # guarantee kill process
        pid = os.getpid()
        os.kill(pid, signal.SIGTERM)

    @classmethod
    def tune(cls, task_function: Callable, tasks_param: list[Sequence[Any]],
             max_active_processes: int = 4):
        processes = []
        semaphore = BoundedSemaphore(max_active_processes)

        with Manager() as manager:
            records = manager.list()
            for i in range(len(tasks_param)):
                process = Process(target=multi_process_tuning._run_task_wrapper,
                                  args=(semaphore, task_function, tasks_param[i], records))
                process.start()
                processes.append(process)

            [process.join() for process in processes]
            return list(records)
