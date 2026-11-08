"""Server: serves the node-graph UI, runs workflows, streams progress over websocket."""
import argparse
import asyncio
import json
import os
import queue
import threading
import time
import traceback
import uuid

from aiohttp import web

from . import nodes
from .gpu_pool import GPUPool

WEB_DIR = os.path.join(nodes.ROOT, "web")


def detect_gpus():
    """Return [(index, short_name)] ordered fastest first. Uses nvidia-smi so the
    main process never creates a CUDA context."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, mem = [x.strip() for x in line.split(",")]
        short = name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
        gpus.append((int(idx), f"gpu{idx}-{short}", int(mem)))
    # bigger VRAM first => 4070 Super (12GB) is primary over 4060 Ti 8GB
    gpus.sort(key=lambda g: -g[2])
    return [(i, n) for i, n, _ in gpus]


class Ctx:
    def __init__(self, server, prompt_id, node_id):
        self.server, self.pool = server, server.pool
        self.prompt_id, self.node_id = prompt_id, node_id

    def log(self, msg):
        self.server.broadcast({"type": "log", "node": self.node_id, "msg": msg})

    def progress(self, p):
        self.server.broadcast({"type": "progress", "node": self.node_id, **p})

    def ui(self, data):
        self.server.broadcast({"type": "executed", "node": self.node_id, "output": data})


class Server:
    def __init__(self, gpus):
        self.clients = set()
        self.loop = None
        self.pool = GPUPool(gpus, on_progress=self._gpu_progress)
        nodes.GPU_CHOICES.extend(self.pool.names)
        self.jobs = queue.Queue()
        threading.Thread(target=self._exec_loop, daemon=True).start()

    # -------------------------------------------------------------- events --
    def broadcast(self, msg):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._send(msg), self.loop)

    async def _send(self, msg):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    def _gpu_progress(self, gpu, p):
        self.broadcast({"type": "gpu_progress", "gpu": gpu, **p})

    # ------------------------------------------------------------ executor --
    def _exec_loop(self):
        while True:
            pid, graph = self.jobs.get()
            t0 = time.time()
            self.broadcast({"type": "status", "msg": "running", "prompt_id": pid})
            try:
                self.execute(pid, graph)
                self.broadcast({"type": "status", "msg": f"done in {time.time() - t0:.1f}s", "prompt_id": pid})
            except Exception as e:
                traceback.print_exc()
                self.broadcast({"type": "error", "msg": str(e), "prompt_id": pid})
            self.broadcast({"type": "executing", "node": None})

    def execute(self, pid, graph):
        """graph: {node_id: {"class": str, "widgets": {...}, "inputs": {name: [src_id, out_idx]}}}"""
        results, order, seen = {}, [], set()

        def visit(nid, stack=()):
            if nid in seen:
                return
            if nid in stack:
                raise ValueError("cycle in graph")
            for src in graph[nid].get("inputs", {}).values():
                visit(str(src[0]), stack + (nid,))
            seen.add(nid)
            order.append(nid)

        outputs = [n for n, d in graph.items() if nodes.NODES[d["class"]].OUTPUT_NODE]
        if not outputs:
            raise ValueError("Add a SaveVideo / PreviewVideo node")
        for n in outputs:
            visit(n)

        for nid in order:
            d = graph[nid]
            cls = nodes.NODES[d["class"]]
            kwargs = dict(d.get("widgets", {}))
            for name, (t, optional) in cls.INPUTS.items():
                link = d.get("inputs", {}).get(name)
                if link is None:
                    if not optional:
                        raise ValueError(f"{d['class']} #{nid}: input '{name}' not connected")
                    continue
                kwargs[name] = results[str(link[0])][int(link[1])]
            self.broadcast({"type": "executing", "node": nid})
            results[nid] = cls().run(Ctx(self, pid, nid), **kwargs)


async def ws_handler(request):
    srv = request.app["srv"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    srv.clients.add(ws)
    await ws.send_json({"type": "gpus", "gpus": srv.pool.info})
    async for _ in ws:
        pass
    srv.clients.discard(ws)
    return ws


async def object_info(request):
    return web.json_response(nodes.object_info())


async def system_info(request):
    return web.json_response({"gpus": request.app["srv"].pool.info, "names": request.app["srv"].pool.names})


async def post_prompt(request):
    body = await request.json()
    pid = uuid.uuid4().hex
    request.app["srv"].jobs.put((pid, body["graph"]))
    return web.json_response({"prompt_id": pid})


async def upload(request):
    reader = await request.multipart()
    field = await reader.next()
    name = f"{uuid.uuid4().hex[:6]}_{os.path.basename(field.filename)}"
    with open(os.path.join(nodes.INPUT_DIR, name), "wb") as f:
        while chunk := await field.read_chunk():
            f.write(chunk)
    return web.json_response({"name": name})


async def view(request):
    kind = request.match_info["kind"]
    d = {"input": nodes.INPUT_DIR, "output": nodes.OUTPUT_DIR}[kind]
    path = os.path.join(d, os.path.basename(request.match_info["name"]))
    if not os.path.exists(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def index(request):
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gpus", default="", help="comma list of CUDA indices to use, e.g. 0,1")
    a = ap.parse_args()

    gpus = detect_gpus()
    if a.gpus:
        keep = {int(x) for x in a.gpus.split(",")}
        gpus = [g for g in gpus if g[0] in keep]
    if not gpus:
        print("WARNING: no NVIDIA GPUs found via nvidia-smi; using a single cuda:0 worker")
        gpus = [(0, "gpu0")]
    print("GPU workers:", ", ".join(n for _, n in gpus))

    srv = Server(gpus)
    app = web.Application(client_max_size=200 * 2**20)
    app["srv"] = srv

    async def on_start(_):
        srv.loop = asyncio.get_running_loop()
    app.on_startup.append(on_start)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/object_info", object_info)
    app.router.add_get("/system_info", system_info)
    app.router.add_post("/prompt", post_prompt)
    app.router.add_post("/upload", upload)
    app.router.add_get("/view/{kind}/{name}", view)
    app.router.add_static("/web", WEB_DIR)
    print(f"Open http://{a.host}:{a.port}")
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
