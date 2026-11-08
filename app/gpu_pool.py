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
        self.single = [n for _, n in gpus]
        specs = list(gpus)
        if len(gpus) > 1:
            # one extra worker that sees both cards, for models split across GPUs (Wan 2.2).
            # Only one "group" (per-card workers vs the dual worker) holds models at a time.
            specs.append((",".join(str(i) for i, _ in gpus[:2]), "dual"))
        for idx, name in specs:
            q = ctx.Queue()
            p = ctx.Process(target=worker_main, args=(idx, name, q, self.result_q), daemon=True)
            p.start()
            self.workers[name] = (p, q)
            self.busy[name] = 0
        self.active_group = None
        threading.Thread(target=self._pump, daemon=True).start()

    @property
    def names(self):
        """Per-card workers (excludes the dual worker)."""
        return list(self.single)

    @property
    def dual(self):
        """Name of the worker that spans both cards (falls back to the only card)."""
        return "dual" if "dual" in self.workers else self.single[0]

    def _claim(self, gpu):
        group = "dual" if gpu == "dual" else "single"
        if group == self.active_group:
            return
        others = [n for n in self.workers if (n == "dual") != (group == "dual")]
        self.active_group = group
        for f in [self._raw_submit("unload", {}, n) for n in others]:
            f.result()

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
        if gpu in (None, "auto"):
            gpu = min(self.names, key=lambda n: self.busy[n])
        self._claim(gpu)
        return self._raw_submit(kind, params, gpu)

    def _raw_submit(self, kind, params, gpu):
        with self._lock:
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
