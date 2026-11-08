"""GPU worker process. One process per physical GPU.

CUDA_VISIBLE_DEVICES is set *before* torch is imported, so inside the worker the
card is always ``cuda:0``. This gives each GPU its own CUDA context, its own
model cache and true parallelism (no GIL sharing) -- which matters on a 4-core
i5-6600K.
"""
import os
import traceback


def worker_main(gpu_index, name: str, task_q, result_q):
    # gpu_index is an int (one card) or "0,1" (the dual worker sees both cards:
    # cuda:0 = primary/4070 Super, cuda:1 = secondary/4060 Ti)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi numbering
    # Offline: load models only from disk, never contact Hugging Face (unless --online)
    if os.environ.get("DUALGPU_ONLINE") != "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import warnings
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch  # noqa: E402  (must come after env var)

    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        result_q.put(("fatal", name, f"PyTorch {torch.__version__} has no CUDA support. Reinstall with: "
                      "pip uninstall -y torch torchvision && pip install torch torchvision "
                      "--index-url https://download.pytorch.org/whl/cu128"))
    cache = {}
    result_q.put(("ready", name, {
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else 0,
    }))

    if torch.cuda.device_count() > 1:
        result_q.put(("ready", name, {
            "name": " + ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())),
            "vram_gb": round(sum(torch.cuda.get_device_properties(i).total_memory
                                 for i in range(torch.cuda.device_count())) / 2**30, 1)}))

    while True:
        job = task_q.get()
        if job is None:
            break
        job_id, kind, params = job
        try:
            fn = TASKS[kind]
            out = fn(torch, cache, params, lambda p: result_q.put(("progress", job_id, p)))
            result_q.put(("done", job_id, out))
        except Exception as e:  # report to the scheduler, keep worker alive
            result_q.put(("error", job_id, f"{e}\n{traceback.format_exc()}"))
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Task implementations
# --------------------------------------------------------------------------- #

def _svd_pipe(torch, cache, model, low_vram):
    key = ("svd", model, low_vram)
    if key not in cache:
        cache.clear()  # one big model per card at a time
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16")
        if low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.unet.enable_forward_chunking()
        cache[key] = pipe
    return cache[key]


def _vae(torch, cache, model):
    key = ("vae", model)
    if key not in cache:
        from diffusers import AutoencoderKLTemporalDecoder
        cache[key] = AutoencoderKLTemporalDecoder.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float16, variant="fp16").to("cuda")
    return cache[key]


def task_svd(torch, cache, p, progress):
    """Image -> video with Stable Video Diffusion.

    If ``latent_only`` is set the clip is returned as latents (saved to disk) so a
    second GPU can do the VAE decode -- the dual-GPU *pipeline* mode.
    """
    from PIL import Image
    pipe = _svd_pipe(torch, cache, p["model"], p.get("low_vram", False))
    image = Image.open(p["image"]).convert("RGB").resize((p["width"], p["height"]))
    steps = p["steps"]

    def cb(_pipe, i, _t, kw):
        progress({"step": i + 1, "steps": steps})
        return kw

    result = pipe(
        image,
        width=p["width"], height=p["height"],
        num_frames=p["frames"], num_inference_steps=steps,
        motion_bucket_id=p["motion"], fps=p["fps"],
        noise_aug_strength=p["noise_aug"],
        min_guidance_scale=p["min_cfg"], max_guidance_scale=p["max_cfg"],
        decode_chunk_size=p.get("decode_chunk", 4),
        generator=torch.manual_seed(p["seed"]),
        output_type="latent" if p.get("latent_only") else "pil",
        callback_on_step_end=cb,
    )
    if p.get("latent_only"):
        torch.save(result.frames.cpu(), p["out"])
        return {"latents": p["out"], "frames": p["frames"]}
    paths = _save_frames(result.frames[0], p["out"])
    return {"frames": paths}


def task_vae_decode(torch, cache, p, progress):
    """Decode SVD latents on this card (the second GPU in pipeline mode)."""
    import numpy as np
    from PIL import Image
    vae = _vae(torch, cache, p["model"])
    latents = torch.load(p["latents"]).to("cuda", torch.float16)
    n = latents.shape[1]
    latents = latents.flatten(0, 1) / vae.config.scaling_factor
    chunk = p.get("decode_chunk", 8)
    frames = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], chunk):
            sl = latents[i:i + chunk]
            frames.append(vae.decode(sl, num_frames=sl.shape[0]).sample.float().cpu())
            progress({"step": min(i + chunk, n), "steps": n})
    video = torch.cat(frames)
    video = ((video / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    paths = _save_frames([Image.fromarray(f) for f in video], p["out"])
    os.remove(p["latents"])
    return {"frames": paths}


def task_upscale(torch, cache, p, progress):
    """Simple GPU frame upscale (bicubic + unsharp) -- used to post-process
    frames from either backend, sharded across both GPUs."""
    import numpy as np
    from PIL import Image
    out = []
    for i, f in enumerate(p["frames"]):
        img = torch.from_numpy(np.array(Image.open(f).convert("RGB"))).cuda().permute(2, 0, 1)[None].float()
        up = torch.nn.functional.interpolate(img, scale_factor=p["scale"], mode="bicubic", align_corners=False)
        blur = torch.nn.functional.avg_pool2d(up, 3, 1, 1)
        up = (up + p.get("sharpen", 0.5) * (up - blur)).clamp(0, 255)
        arr = up[0].permute(1, 2, 0).byte().cpu().numpy()
        path = os.path.join(p["out"], f"up_{p.get('start', 0) + i:06d}.png")
        os.makedirs(p["out"], exist_ok=True)
        Image.fromarray(arr).save(path)
        out.append(path)
        progress({"step": i + 1, "steps": len(p["frames"])})
    return {"frames": out}


def task_interpolate(torch, cache, p, progress):
    """Linear frame blending on GPU to raise frame rate (x2/x3/x4)."""
    import numpy as np
    from PIL import Image
    os.makedirs(p["out"], exist_ok=True)
    frames, mult, out = p["frames"], p["multiplier"], []
    load = lambda f: torch.from_numpy(np.array(Image.open(f).convert("RGB"))).cuda().float()
    prev = load(frames[0])
    for i in range(1, len(frames) + 1):
        out.append(prev)
        if i == len(frames):
            break
        nxt = load(frames[i])
        for k in range(1, mult):
            t = k / mult
            out.append(prev * (1 - t) + nxt * t)
        prev = nxt
        progress({"step": i, "steps": len(frames) - 1})
    paths = []
    for j, f in enumerate(out):
        path = os.path.join(p["out"], f"{p.get('prefix', 'f')}_{j:05d}.png")
        Image.fromarray(f.clamp(0, 255).byte().cpu().numpy()).save(path)
        paths.append(path)
    return {"frames": paths}


def _save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, im in enumerate(frames):
        path = os.path.join(out_dir, f"frame_{i:05d}.png")
        im.save(path)
        paths.append(path)
    return paths


def task_unload(torch, cache, p, progress):
    cache.clear()
    return {}


TASKS = {
    "unload": task_unload,
    "svd": task_svd,
    "vae_decode": task_vae_decode,
    "upscale": task_upscale,
    "interpolate": task_interpolate,
}


from .wan import task_wan  # noqa: E402
TASKS["wan"] = task_wan
