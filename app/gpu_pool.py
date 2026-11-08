"""Manages one worker process per GPU and routes jobs to them."""
import itertools
import multiprocessing as mp
import threading
from concurrent.futures import Future

from .worker import worker_main


class GPUPool:
    def __init__(self, gpus, on_progress=None):
        """gpus: list of (cuda_index, name) e.g. [(0, "4070S"), (1, "4060Ti")]"""
        ctx = mp.get_context("spawn")
        self.result_q = ctx.Queue()
        self.workers = {}
        self.info = {}
        self.busy = {}
        self.futures = {}
        self.on_progress = on_progress or (lambda *a: None)
        self._ids = itertools.count()
        self._lock = threading.Lock()
        for idx, name in gpus:
            q = ctx.Queue()
            p = ctx.Process(target=worker_main, args=(idx, name, q, self.result_q), daemon=True)
            p.start()
            self.workers[name] = (p, q)
            self.busy[name] = 0
        threading.Thread(target=self._pump, daemon=True).start()

    @property
    def names(self):
        return list(self.workers)

    def _pump(self):
        while True:
            kind, key, payload = self.result_q.get()
            if kind == "ready":
                self.info[key] = payload
                continue
            fut, gpu = self.futures.get(key, (None, None))
            if kind == "progress":
                self.on_progress(gpu, payload)
                continue
            with self._lock:
                self.futures.pop(key, None)
                self.busy[gpu] -= 1
            if kind == "done":
                fut.set_result(payload)
            else:
                fut.set_exception(RuntimeError(f"[{gpu}] {payload}"))

    def submit(self, kind, params, gpu=None) -> Future:
        """Submit to a named GPU, or 'auto' = least busy (ties -> first/fastest)."""
        with self._lock:
            if gpu in (None, "auto"):
                gpu = min(self.names, key=lambda n: self.busy[n])
            if gpu not in self.workers:
                raise ValueError(f"Unknown GPU '{gpu}'. Have: {self.names}")
            job_id = next(self._ids)
            fut = Future()
            self.futures[job_id] = (fut, gpu)
            self.busy[gpu] += 1
        self.workers[gpu][1].put((job_id, kind, params))
        return fut

    def shutdown(self):
        for p, q in self.workers.values():
            q.put(None)
