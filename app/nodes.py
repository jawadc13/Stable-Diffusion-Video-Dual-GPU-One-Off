"""Node definitions. Each node declares typed inputs/outputs and widgets; the
web UI builds itself from ``/object_info`` exactly like ComfyUI does.

Socket types: IMAGE (path to png), FRAMES (list of png paths), VIDEO (mp4 path), STRING
"""
import os
import time
import uuid

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "input")
OUTPUT_DIR = os.path.join(ROOT, "output")
TEMP_DIR = os.path.join(ROOT, "temp")
for d in (INPUT_DIR, OUTPUT_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

GPU_CHOICES = ["auto"]  # filled with real names at startup
NODES = {}


def node(cls):
    NODES[cls.__name__] = cls
    return cls


def _tmp(prefix):
    return os.path.join(TEMP_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}")


def _shard(items, n):
    k, r = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        s = k + (1 if j < r else 0)
        out.append((i, items[i:i + s]))
        i += s
    return [o for o in out if o[1]]


class Node:
    CATEGORY = "misc"
    INPUTS = {}      # name -> (type, optional)
    WIDGETS = {}     # name -> {"type": int|float|str|combo|text|image, default, min, max, options}
    OUTPUTS = []     # [(name, type)]
    OUTPUT_NODE = False


# ----------------------------------------------------------------- inputs --
@node
class LoadImage(Node):
    CATEGORY = "input"
    WIDGETS = {"image": {"type": "image", "default": ""}}
    OUTPUTS = [("IMAGE", "IMAGE")]

    def run(self, ctx, image):
        path = os.path.join(INPUT_DIR, os.path.basename(image))
        if not image or not os.path.exists(path):
            raise ValueError("LoadImage: upload an image first")
        return (path,)


@node
class TextPrompt(Node):
    CATEGORY = "input"
    WIDGETS = {"text": {"type": "text", "default": "a cinematic drone shot over misty mountains at sunrise"}}
    OUTPUTS = [("STRING", "STRING")]

    def run(self, ctx, text):
        return (text,)


# ------------------------------------------------------------- MiniMax API --
@node
class MiniMaxVideo(Node):
    """(Optional, needs a paid API key) Text/Image -> video via MiniMax Hailuo cloud API. Runs on MiniMax's servers;
    the result is pulled back as frames so your local GPUs can post-process it."""
    CATEGORY = "generate"
    INPUTS = {"prompt": ("STRING", False), "first_frame": ("IMAGE", True)}
    WIDGETS = {
        "model": {"type": "combo", "default": "MiniMax-Hailuo-02",
                  "options": ["MiniMax-Hailuo-02", "T2V-01-Director", "I2V-01-Director", "I2V-01", "T2V-01"]},
        "duration": {"type": "combo", "default": "6", "options": ["6", "10"]},
        "resolution": {"type": "combo", "default": "768P", "options": ["512P", "768P", "1080P"]},
        "prompt_optimizer": {"type": "combo", "default": "true", "options": ["true", "false"]},
        "api_key": {"type": "str", "default": "", "password": True},
        "api_base": {"type": "str", "default": "https://api.minimax.io"},
    }
    OUTPUTS = [("VIDEO", "VIDEO"), ("FRAMES", "FRAMES")]

    def run(self, ctx, prompt, model, duration, resolution, prompt_optimizer, api_key, api_base,
            first_frame=None):
        import base64
        key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not key:
            raise ValueError("MiniMaxVideo: set api_key widget or MINIMAX_API_KEY env var")
        h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": model, "prompt": prompt, "duration": int(duration),
                "resolution": resolution, "prompt_optimizer": prompt_optimizer == "true"}
        if first_frame:
            with open(first_frame, "rb") as f:
                body["first_frame_image"] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        r = requests.post(f"{api_base}/v1/video_generation", headers=h, json=body, timeout=120).json()
        task_id = r.get("task_id")
        if not task_id:
            raise RuntimeError(f"MiniMax error: {r.get('base_resp', r)}")
        ctx.log(f"MiniMax task {task_id} submitted")
        t0 = time.time()
        while True:
            time.sleep(10)
            q = requests.get(f"{api_base}/v1/query/video_generation", headers=h,
                             params={"task_id": task_id}, timeout=60).json()
            status = q.get("status")
            ctx.progress({"text": f"MiniMax: {status} ({int(time.time() - t0)}s)"})
            if status == "Success":
                file_id = q["file_id"]
                break
            if status == "Fail":
                raise RuntimeError(f"MiniMax generation failed: {q}")
        f = requests.get(f"{api_base}/v1/files/retrieve", headers=h,
                         params={"file_id": file_id}, timeout=60).json()
        url = f["file"]["download_url"]
        path = os.path.join(OUTPUT_DIR, f"minimax_{task_id}.mp4")
        with requests.get(url, stream=True, timeout=300) as resp, open(path, "wb") as out:
            for chunk in resp.iter_content(1 << 20):
                out.write(chunk)
        return (path, _video_to_frames(path))


def _video_to_frames(path):
    import imageio.v3 as iio
    from PIL import Image
    d = _tmp("vidframes")
    os.makedirs(d)
    paths = []
    for i, fr in enumerate(iio.imiter(path)):
        p = os.path.join(d, f"frame_{i:05d}.png")
        Image.fromarray(fr).save(p)
        paths.append(p)
    return paths


# ------------------------------------------------ Wan 2.2 (open, no key) --
QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "bf16"]


@node
class WanVideo(Node):
    """Wan 2.2 text/image -> video. Open weights (Apache-2.0), runs 100% locally, no API key.

    A14B (best quality): high-noise expert on the primary GPU, low-noise expert on the
    secondary GPU, VAE decode on the secondary. TI2V-5B (faster): transformer on primary,
    VAE on secondary. Text encoder runs on CPU / system RAM. Connect an image for
    image-to-video. Frames are rounded to 4n+1."""
    CATEGORY = "generate"
    INPUTS = {"prompt": ("STRING", False), "image": ("IMAGE", True)}
    WIDGETS = {
        "model": {"type": "combo", "default": "A14B", "options": ["A14B", "TI2V-5B"]},
        "resolution": {"type": "combo", "default": "832x480",
                       "options": ["832x480", "480x832", "1280x720", "720x1280", "1280x704", "704x1280", "640x640"]},
        "frames": {"type": "int", "default": 81, "min": 5, "max": 161, "step": 4},
        "steps": {"type": "int", "default": 30, "min": 4, "max": 80},
        "guidance": {"type": "float", "default": 4.0, "min": 1, "max": 10, "step": 0.5},
        "guidance_2": {"type": "float", "default": 3.0, "min": 1, "max": 10, "step": 0.5},
        "seed": {"type": "int", "default": 42, "min": 0, "max": 2**31 - 1},
        "quant_high": {"type": "combo", "default": "Q4_K_M", "options": QUANTS},
        "quant_low": {"type": "combo", "default": "Q3_K_M", "options": QUANTS},
        "negative_prompt": {"type": "text", "default":
                            "blurry, low quality, worst quality, distorted, deformed, static, "
                            "watermark, text, subtitles, jpeg artifacts, extra fingers, bad anatomy"},
    }
    OUTPUTS = [("FRAMES", "FRAMES")]

    def run(self, ctx, prompt, model, resolution, frames, steps, guidance, guidance_2, seed,
            quant_high, quant_low, negative_prompt, image=None):
        w, h = map(int, resolution.split("x"))
        frames = (frames - 1) // 4 * 4 + 1
        variant = "5B" if model == "TI2V-5B" else ("A14B-I2V" if image else "A14B-T2V")
        if variant == "5B" and quant_high not in ("Q8_0", "bf16"):
            ctx.log("TI2V-5B: tip -- Q8_0 fits the 4070 Super comfortably and keeps full quality")
        ctx.log(f"Wan 2.2 {variant} {w}x{h} {frames}f on {ctx.pool.dual} "
                "(first run downloads the model)")
        fut = ctx.pool.submit("wan", dict(
            variant=variant, prompt=prompt, negative_prompt=negative_prompt, image=image,
            width=w, height=h, frames=frames, steps=steps, guidance=guidance,
            guidance_2=guidance_2, seed=seed, quant_high=quant_high, quant_low=quant_low,
            out=_tmp("wan")), ctx.pool.dual)
        return (fut.result()["frames"],)


# ------------------------------------------------------ Local SVD dual GPU --
@node
class SVDDualGPU(Node):
    """Stable Video Diffusion image->video using both GPUs.

    modes
      parallel : each GPU renders whole clips; `batch` clips/seeds split between cards
      pipeline : GPU A denoises (UNet), GPU B VAE-decodes; clips overlap so both stay busy
      single   : only `primary_gpu`
    """
    CATEGORY = "generate"
    INPUTS = {"image": ("IMAGE", False)}
    WIDGETS = {
        "model": {"type": "combo", "default": "stabilityai/stable-video-diffusion-img2vid-xt",
                  "options": ["stabilityai/stable-video-diffusion-img2vid-xt",
                              "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                              "stabilityai/stable-video-diffusion-img2vid"]},
        "mode": {"type": "combo", "default": "parallel", "options": ["parallel", "pipeline", "single"]},
        "primary_gpu": {"type": "combo", "default": "auto", "options": GPU_CHOICES},
        "width": {"type": "int", "default": 1024, "min": 256, "max": 1536, "step": 64},
        "height": {"type": "int", "default": 576, "min": 256, "max": 1536, "step": 64},
        "frames": {"type": "int", "default": 25, "min": 8, "max": 50},
        "steps": {"type": "int", "default": 25, "min": 1, "max": 100},
        "motion": {"type": "int", "default": 127, "min": 1, "max": 255},
        "fps": {"type": "int", "default": 7, "min": 1, "max": 60},
        "noise_aug": {"type": "float", "default": 0.02, "min": 0, "max": 1, "step": 0.01},
        "min_cfg": {"type": "float", "default": 1.0, "min": 0, "max": 10, "step": 0.1},
        "max_cfg": {"type": "float", "default": 3.0, "min": 0, "max": 10, "step": 0.1},
        "seed": {"type": "int", "default": 42, "min": 0, "max": 2**31 - 1},
        "batch": {"type": "int", "default": 2, "min": 1, "max": 32},
        "low_vram_8gb": {"type": "combo", "default": "auto", "options": ["auto", "on", "off"]},
    }
    OUTPUTS = [("FRAMES", "FRAMES")]

    def run(self, ctx, image, model, mode, primary_gpu, width, height, frames, steps, motion,
            fps, noise_aug, min_cfg, max_cfg, seed, batch, low_vram_8gb):
        pool = ctx.pool
        gpus = pool.names
        base = dict(model=model, image=image, width=width, height=height, frames=frames,
                    steps=steps, motion=motion, fps=fps, noise_aug=noise_aug,
                    min_cfg=min_cfg, max_cfg=max_cfg)

        def lowvram(g):
            if low_vram_8gb != "auto":
                return low_vram_8gb == "on"
            return pool.info.get(g, {}).get("vram_gb", 16) < 11

        primary = gpus[0] if primary_gpu == "auto" else primary_gpu
        secondary = next((g for g in gpus if g != primary), primary)
        futures = []
        if mode == "single" or len(gpus) == 1:
            for b in range(batch):
                futures.append(pool.submit("svd", dict(base, seed=seed + b, out=_tmp("svd"),
                                                       low_vram=lowvram(primary)), primary))
        elif mode == "parallel":
            # weight by VRAM so the bigger card takes more clips when uneven
            for b in range(batch):
                futures.append(pool.submit("svd", dict(base, seed=seed + b, out=_tmp("svd"),
                                                       low_vram=lowvram(primary)
                                                       if b % 2 == 0 else lowvram(secondary)),
                                           primary if b % 2 == 0 else secondary))
        else:  # pipeline
            lat = [pool.submit("svd", dict(base, seed=seed + b, latent_only=True,
                                           out=_tmp("lat") + ".pt", low_vram=lowvram(primary)), primary)
                   for b in range(batch)]
            for f in lat:  # decode each clip as soon as its latents are ready
                r = f.result()
                futures.append(pool.submit("vae_decode", dict(model=model, latents=r["latents"],
                                                              out=_tmp("dec")), secondary))
        allframes = []
        for i, f in enumerate(futures):
            allframes += f.result()["frames"]
            ctx.log(f"SVD clip {i + 1}/{len(futures)} done")
        return (allframes,)


# ------------------------------------------------------------ frame tools --
@node
class LastFrame(Node):
    """Take the last frame of a clip -> IMAGE (chain MiniMax -> SVD, or extend clips)."""
    CATEGORY = "frames"
    INPUTS = {"frames": ("FRAMES", False)}
    OUTPUTS = [("IMAGE", "IMAGE")]

    def run(self, ctx, frames):
        return (frames[-1],)


@node
class ConcatFrames(Node):
    CATEGORY = "frames"
    INPUTS = {"a": ("FRAMES", False), "b": ("FRAMES", False)}
    OUTPUTS = [("FRAMES", "FRAMES")]

    def run(self, ctx, a, b):
        return (a + b,)


@node
class UpscaleDualGPU(Node):
    """Upscale frames, split across both GPUs."""
    CATEGORY = "frames"
    INPUTS = {"frames": ("FRAMES", False)}
    WIDGETS = {"scale": {"type": "float", "default": 2.0, "min": 1, "max": 4, "step": 0.5},
               "sharpen": {"type": "float", "default": 0.5, "min": 0, "max": 2, "step": 0.1}}
    OUTPUTS = [("FRAMES", "FRAMES")]

    def run(self, ctx, frames, scale, sharpen):
        out = _tmp("up")
        futs = [ctx.pool.submit("upscale", dict(frames=chunk, start=s, scale=scale, sharpen=sharpen, out=out), g)
                for g, (s, chunk) in zip(ctx.pool.names, _shard(frames, len(ctx.pool.names)))]
        res = []
        for f in futs:
            res += f.result()["frames"]
        return (res,)


@node
class InterpolateDualGPU(Node):
    """Raise frame-rate by blending, split across both GPUs."""
    CATEGORY = "frames"
    INPUTS = {"frames": ("FRAMES", False)}
    WIDGETS = {"multiplier": {"type": "int", "default": 2, "min": 2, "max": 4}}
    OUTPUTS = [("FRAMES", "FRAMES")]

    def run(self, ctx, frames, multiplier):
        out = _tmp("interp")
        names = ctx.pool.names
        shards = _shard(frames, len(names))
        futs = []
        for j, (g, (start, chunk)) in enumerate(zip(names, shards)):
            # overlap one frame into the next shard so the seam gets in-betweens too
            seg = frames[start:start + len(chunk) + (1 if j < len(shards) - 1 else 0)]
            futs.append((j < len(shards) - 1, ctx.pool.submit(
                "interpolate", dict(frames=seg, multiplier=multiplier, out=out, prefix=f"s{j}"), g)))
        res = []
        for overlapped, f in futs:
            r = f.result()["frames"]
            res += r[:-1] if overlapped else r
        return (res,)


# ---------------------------------------------------------------- outputs --
@node
class SaveVideo(Node):
    CATEGORY = "output"
    OUTPUT_NODE = True
    INPUTS = {"frames": ("FRAMES", False)}
    WIDGETS = {"fps": {"type": "int", "default": 8, "min": 1, "max": 120},
               "filename_prefix": {"type": "str", "default": "dualgpu"},
               "quality": {"type": "int", "default": 8, "min": 1, "max": 10}}
    OUTPUTS = [("VIDEO", "VIDEO")]

    def run(self, ctx, frames, fps, filename_prefix, quality):
        import imageio
        import numpy as np
        from PIL import Image
        if not frames:
            raise ValueError("SaveVideo: no frames")
        name = f"{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(OUTPUT_DIR, name)
        size = Image.open(frames[0]).size
        with imageio.get_writer(path, fps=fps, quality=quality, macro_block_size=8) as w:
            for f in frames:
                im = Image.open(f).convert("RGB")
                if im.size != size:
                    im = im.resize(size)
                w.append_data(np.array(im))
        ctx.ui({"video": name})
        return (path,)


@node
class PreviewVideo(Node):
    CATEGORY = "output"
    OUTPUT_NODE = True
    INPUTS = {"video": ("VIDEO", False)}

    def run(self, ctx, video):
        import shutil
        name = os.path.basename(video)
        if os.path.dirname(os.path.abspath(video)) != OUTPUT_DIR:
            shutil.copy(video, os.path.join(OUTPUT_DIR, name))
        ctx.ui({"video": name})
        return ()


def object_info():
    info = {}
    for name, cls in NODES.items():
        info[name] = {
            "category": cls.CATEGORY,
            "description": (cls.__doc__ or "").strip(),
            "inputs": {k: {"type": t, "optional": o} for k, (t, o) in cls.INPUTS.items()},
            "widgets": cls.WIDGETS,
            "outputs": cls.OUTPUTS,
            "output_node": cls.OUTPUT_NODE,
        }
    return info
