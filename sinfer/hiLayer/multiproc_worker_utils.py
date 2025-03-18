import asyncio
import multiprocessing
import os
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO,
                    TypeVar, Union)

import sinfer.envs as envs
from sinfer.logger import init_logger

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# ANSI颜色代码，用于输出带颜色的日志信息
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2

mp_method = envs.SINFER_WORKER_MULTIPROC_METHOD
mp = multiprocessing.get_context(mp_method)


@dataclass
class Result(Generic[T]):
    """节点执行任务的结果"""
    task_id: uuid.UUID               # 任务的唯一ID
    value: Optional[T] = None        # 任务执行的结果值
    exception: Optional[BaseException] = None  # 如果任务出错，则记录异常信息

class ResultFuture(threading.Event, Generic[T]):
    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    # 设置任务结果
    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    # 获取任务结果
    def get(self) -> T:
        self.wait()
        assert self.result is not None
        # 如果有异常，抛出
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value  # 返回结果值

def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if result.exception is not None:
        loop.call_soon_threadsafe(future.set_exception, result.exception)
    else:
        loop.call_soon_threadsafe(future.set_result, result.value)

class ResultHandler(threading.Thread):
    """用于处理所有节点返回的结果(后台线程)"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = mp.Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    # 后台线程的主循环，用于处理结果
    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        # 确保所有等待任务收到异常
        for task_id, future in self.tasks.items():
            _set_future_result(
                future,
                Result(task_id=task_id,
                       exception=ChildProcessError("worker died")))

    def close(self):
        self.result_queue.put(_TERMINATE)

class WorkerMonitor(threading.Thread):
    """监视工作节点的状态(后台线程)"""

    def __init__(self, workers: List['ProcessWorkerWrapper'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers # 监视的工作进程列表
        self.result_handler = result_handler
        self._close = False

    # 后台线程的主循环 监视节点进程是否退出
    def run(self) -> None:
        dead_sentinels = wait([w.process.sentinel for w in self.workers]) # 等待进程退出
        if not self._close:
            self._close = True
            for worker in self.workers:
                process = worker.process
                if process.sentinel in dead_sentinels:
                    process.join(JOIN_TIMEOUT_S)
                if process.exitcode is not None and process.exitcode != 0:
                    logger.error("Worker %s pid %s died, exit code: %s",
                                 process.name, process.pid, process.exitcode)
            # 清理剩余的worker
            logger.info("Killing local SInfer worker processes")
            for worker in self.workers:
                worker.kill_worker()
            # 必须在worker任务队列全部关闭后执行
            self.result_handler.close()

        for worker in self.workers:
            worker.process.join(JOIN_TIMEOUT_S)

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local SInfer worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        self.result_handler.close()

class ProcessWorkerWrapper:
    """本地进程的封装类，用于 sinfer.worker.Worker 实现单节点多MLU的张量并行"""
    def __init__(self, result_handler: ResultHandler,
                 worker_factory: Callable[[], Any]) -> None:
        self._task_queue = mp.Queue() # 任务队列
        self.result_queue = result_handler.result_queue  # 结果队列
        self.tasks = result_handler.tasks 
        # 创建一个新的工作进程
        self.process: BaseProcess = mp.Process( 
            target=_run_worker_process,
            name="SinferWorkerProcess", # 进程的名称
            kwargs=dict(  
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                result_queue=self.result_queue,
            ),              # 给目标函数传递的参数
            daemon=True) # 设置进程为守护进程

        self.process.start()

    # 将任务放入队列
    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future  # 将任务 ID 和 Future 映射
        try:
            self._task_queue.put((task_id, method, args, kwargs))  # 将任务的详细信息放入任务队列
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: str, *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.process.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.process.kill()

def _run_worker_process(
    worker_factory: Callable[[], Any],
    task_queue: Queue,
    result_queue: Queue,
) -> None:
    process_name = mp.current_process().name # 获取当前进程名称
    pid = os.getpid()  # 获取当前进程 ID
    _add_prefix(sys.stdout, process_name, pid)  # 修改 stdout
    _add_prefix(sys.stderr, process_name, pid)

    # 使用工厂方法创建 Worker 实例
    worker = worker_factory()
    del worker_factory

    # 开始处理task_queue中的任务
    logger.info("Worker ready; awaiting tasks")
    try:
        # 从任务队列中读取任务，直到收到终止信号
        for items in iter(task_queue.get, _TERMINATE):
            output = None
            exception = None
            task_id, method, args, kwargs = items # 解包任务参数
            try:
                # 获取 Worker 中的方法并执行
                executor = getattr(worker, method)
                output = executor(*args, **kwargs)
            except BaseException as e:
                tb = traceback.format_exc()
                logger.error(
                    "Exception in worker %s while processing method %s: %s, %s",
                    process_name, method, e, tb)
                exception = e
            result_queue.put(
                Result(task_id=task_id, value=output, exception=exception))  # 将任务执行结果（包括输出和异常）放入结果队列
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    logger.info("Worker exiting")

def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """为输出流的每行添加进程标识前缀"""
    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx]) # 写入当前行内容
            if next_idx == len(s): # 如果到字符串末尾，则设置新行标记
                file.start_new_line = True 
                return
            file_write(prefix)  # 为下一行添加前缀
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # 清除新行标记

    file.start_new_line = True
    file.write = write_with_prefix  # 替换原始的 write 方法
