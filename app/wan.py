"""Wan 2.2 (Alibaba, Apache-2.0, open weights -- no API key) split across two GPUs.

Runs inside the *dual* worker, which sees both cards:
    cuda:0 = primary   (RTX 4070 Super 12 GB)
    cuda:1 = secondary (RTX 4060 Ti)

A14B  : Mixture-of-Experts. The high-noise expert (early steps, layout/motion) lives on
        cuda:0, the low-noise expert (late steps, detail) lives on cuda:1. Latents hop
        across PCIe once per step (a few MB -- negligible). VAE decode also on cuda:1.
TI2V-5B: transformer on cuda:0, VAE (encode + decode) on cuda:1.
Both  : UMT5-XXL text encoder runs on the CPU in fp32 only while a new prompt is encoded,
        then is freed (keeps RAM for the streamed experts); embeddings are cached so changing only the seed doesn't re-encode.
"""
import os

REPOS = {
    "A14B-T2V": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "A14B-I2V": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}
GGUF_REPOS = {
    "A14B-T2V": "QuantStack/Wan2.2-T2V-A14B-GGUF",
    "A14B-I2V": "QuantStack/Wan2.2-I2V-A14B-GGUF",
    "5B": "QuantStack/Wan2.2-TI2V-5B-GGUF",
}


def _model_dirs():
    """Where to look for local .gguf files: ./models first, then the existing HF cache."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
    return [os.path.join(root, "models"), hf]


def _find_gguf(repo, hint, quant):
    """Pick a local .gguf whose path contains `hint` (HighNoise/LowNoise) and `quant`.
    Offline by default -- only downloads when the server was started with --online."""
    want = repo.split("/")[-1].lower().replace("-gguf", "")
    hits = []
    for d in _model_dirs():
        for dp, _, files in os.walk(d):
            for f in files:
                full = os.path.join(dp, f)
                low = full.lower()
                if f.endswith(".gguf") and quant.lower() in f.lower() and hint.lower() in low \
                        and (want in low or want.replace("-", "_") in low or want in f.lower()):
                    hits.append(full)
    if hits:
        return sorted(hits, key=len)[0]
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(
            f"No local {quant} {hint or ''} .gguf for {repo}. Put it in ./models, or start the app "
            f"once with --online to download it.")
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(repo)
             if f.endswith(".gguf") and quant.lower() in f.lower() and hint.lower() in f.lower()]
    if not files:
        raise FileNotFoundError(f"No {quant} {hint} gguf found in {repo}")
    return hf_hub_download(repo, sorted(files, key=len)[0])


def _load_transformer(torch, repo, sub, quant, gguf_repo, hint):
    from diffusers import WanTransformer3DModel
    if quant == "bf16":
        return WanTransformer3DModel.from_pretrained(repo, subfolder=sub, torch_dtype=torch.bfloat16)
    from diffusers import GGUFQuantizationConfig
    return WanTransformer3DModel.from_single_file(
        _find_gguf(gguf_repo, hint, quant),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=repo, subfolder=sub, torch_dtype=torch.bfloat16)


def _on_device_cls(torch):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    def mv(x, d):
        if torch.is_tensor(x):
            return x.to(d)
        if isinstance(x, (tuple, list)):
            return type(x)(mv(i, d) for i in x)
        if isinstance(x, dict):
            return {k: mv(v, d) for k, v in x.items()}
        return x

    class OnDevice(torch.nn.Module):
        """Runs the wrapped module on `dev`; inputs/outputs are moved to/from `home`."""

        def __init__(self, m, dev, home):
            super().__init__()
            self.m, self.dev, self.home = m, torch.device(dev), torch.device(home)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.m, name)

        def forward(self, *a, **k):
            out = self.m(*mv(a, self.dev), **mv(k, self.dev))
            return mv(out, self.home) if isinstance(out, (tuple, list)) or torch.is_tensor(out) \
                else type(out)(**mv(dict(out), self.home))

        # VAE encode (used by image-to-video conditioning)
        def encode(self, x, *a, **k):
            r = self.m.encode(x.to(self.dev), *a, **k)
            dist = r.latent_dist if hasattr(r, "latent_dist") else r[0]
            return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(dist.parameters.to(self.home)))

    return OnDevice


LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"


def _find_lightning(variant, which):
    """Locate the 4-step Lightning LoRA (high_noise / low_noise) locally, else download if --online."""
    kind = "i2v" if variant == "A14B-I2V" else "t2v"
    def ok(path):
        l = path.lower().replace("\\", "/")
        return (l.endswith(".safetensors") and f"{kind}-a14b-4steps" in l and f"{which}_model" in l)
    hits = []
    for d in _model_dirs():
        for dp, _, files in os.walk(d):
            hits += [os.path.join(dp, f) for f in files if ok(os.path.join(dp, f))]
    if hits:
        return sorted(hits)[-1]  # latest version (V1.1 > V1)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(
            "Lightning LoRA not found locally. Start the app once with --online to download it "
            f"from {LIGHTNING_REPO} (~1.2 GB), or put the high_noise_model/low_noise_model "
            "safetensors in ./models/Wan2.2-" + kind.upper() + "-A14B-4steps-lora/")
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(LIGHTNING_REPO) if ok(f))
    if not files:
        raise FileNotFoundError(f"No {kind} 4-step {which} LoRA in {LIGHTNING_REPO}")
    return hf_hub_download(LIGHTNING_REPO, files[-1])


def _load_lightning(pipe, variant):
    """Wan2.2-Lightning: distilled 4-step LoRAs (one per expert). Runs at CFG 1, so each
    step is a single forward pass instead of two -> ~15x fewer passes than 30 steps + CFG."""
    hi, lo = _find_lightning(variant, "high_noise"), _find_lightning(variant, "low_noise")
    print(f"[wan] Lightning LoRAs:\n  {hi}\n  {lo}", flush=True)
    pipe.load_lora_weights(hi, adapter_name="lightning")
    pipe.load_lora_weights(lo, adapter_name="lightning_2", load_into_transformer_2=True)


def _set_attention(pipe, mode):
    """SageAttention (8-bit attention kernels) is ~2x faster than PyTorch SDPA on RTX 40.
    'auto' uses it when the sageattention package is installed, else PyTorch default."""
    if mode == "native":
        return
    backend = {"auto": "sage", "sage": "sage", "flash": "flash"}[mode]
    for t in (pipe.transformer, getattr(pipe, "transformer_2", None)):
        if t is None:
            continue
        try:
            t.set_attention_backend(backend)
        except Exception as e:
            if mode != "auto":
                raise RuntimeError(f"attention backend '{backend}' unavailable: {e}")
            print(f"[wan] SageAttention not available ({type(e).__name__}); using PyTorch SDPA. "
                  "pip install triton-windows sageattention to enable.", flush=True)
            return
    print(f"[wan] attention backend: {backend}", flush=True)


def _build(torch, cache, p):
    from diffusers import WanImageToVideoPipeline, WanPipeline
    from transformers import UMT5EncoderModel

    variant = p["variant"]
    repo, gguf = REPOS[variant], GGUF_REPOS[variant]
    speed = p.get("speed", "off")
    key = ("wan", variant, p["quant_high"], p["quant_low"], bool(p.get("image")), speed,
           p.get("attention", "auto"))
    if cache.get("key") == key:
        return cache["pipe"], cache["te"]
    cache.clear()
    dual = torch.cuda.device_count() > 1
    d0, d1 = "cuda:0", ("cuda:1" if dual else "cuda:0")
    OnDevice = _on_device_cls(torch)

    # Text encoder is NOT kept loaded: it's ~22 GB in fp32 and would push the streamed
    # experts into the pagefile. It's loaded only when the prompt changes (see task_wan).
    te = None
    kw = {"text_encoder": None}
    if variant == "5B":
        kw["transformer"] = _load_transformer(torch, repo, "transformer", p["quant_high"], gguf, "")
    else:
        kw["transformer"] = _load_transformer(torch, repo, "transformer", p["quant_high"], gguf, "HighNoise")
        kw["transformer_2"] = _load_transformer(torch, repo, "transformer_2", p["quant_low"], gguf, "LowNoise")

    cls = WanImageToVideoPipeline if (variant == "A14B-I2V" or p.get("image")) else WanPipeline
    pipe = cls.from_pretrained(repo, torch_dtype=torch.bfloat16, **kw)
    # Pin the pipeline's "execution device" to the primary card even though the
    # text encoder sits on the CPU.
    pipe.__class__ = type(cls.__name__ + "Dual", (cls,),
                          {"_execution_device": property(lambda s: torch.device(d0))})

    if speed == "lightning-4step" and variant != "5B":
        _load_lightning(pipe, variant)
    _set_attention(pipe, p.get("attention", "auto"))

    _place(torch, cache, pipe.transformer, d0, p.get("offload_high", "auto"), "high")
    if getattr(pipe, "transformer_2", None) is not None:
        low = OnDevice(pipe.transformer_2, d1, d0)
        object.__setattr__(pipe, "transformer_2", low)
        _place(torch, cache, low.m, d1, p.get("offload_low", "auto"), "low")
    pipe.vae.to(d1)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    vae_raw = pipe.vae
    object.__setattr__(pipe, "vae", OnDevice(vae_raw, d1, d0))
    if getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder.to(d0)
    cache.update(key=key, pipe=pipe, te=te, vae=vae_raw, d1=d1, embeds={}, repo=repo)
    return pipe, te


def _place(torch, cache, model, dev, mode, tag):
    """Keep `model` resident on `dev` only if weights + ~3 GB of activations fit in the
    card's *currently free* VRAM (the desktop/DWM eats some); otherwise stream it from RAM."""
    weights = sum(t.numel() * t.element_size() for t in model.parameters())
    weights += sum(t.numel() * t.element_size() for t in model.buffers())
    free = torch.cuda.mem_get_info(torch.device(dev))[0]
    need = weights + int(3.0 * 2**30)
    print(f"[wan] {tag}-noise expert {weights / 2**30:.1f} GB, {dev} free {free / 2**30:.1f} GB "
          f"(need ~{need / 2**30:.1f} GB to keep it resident)", flush=True)
    if mode == "always" or (mode == "auto" and need > free):
        _stream(torch, cache, model, dev, tag)
        return
    try:
        model.to(dev)
    except torch.cuda.OutOfMemoryError:
        if mode == "never":
            raise
        _stream(torch, cache, model, dev, tag)


def _stream(torch, cache, model, dev, tag):
    """Group offload: weights stay in system RAM, blocks are copied to the card just in
    time on a side CUDA stream so the copy overlaps compute. Costs a few seconds per
    step over PCIe 3.0 x8, but needs only ~1-2 GB of VRAM for weights."""
    if cache.get(f"{tag}_streamed"):
        return
    print(f"[wan] streaming {tag}-noise expert from system RAM onto {dev}", flush=True)
    model.to("cpu")
    torch.cuda.empty_cache()
    model.enable_group_offload(onload_device=torch.device(dev), offload_device=torch.device("cpu"),
                               offload_type="block_level", num_blocks_per_group=4, use_stream=True)
    cache[f"{tag}_streamed"] = True


def _embeds(torch, cache, pipe, p, progress):
    """Prompt embeddings on CPU (cached). The text encoder is loaded, used and freed."""
    ek = (p["prompt"], p["negative_prompt"])
    if ek not in cache["embeds"]:
        import gc

        from transformers import UMT5EncoderModel
        progress({"text": "encoding prompt on CPU (loads text encoder, then frees it)..."})
        te = UMT5EncoderModel.from_pretrained(cache["repo"], subfolder="text_encoder",
                                              torch_dtype=torch.float32, low_cpu_mem_usage=True)
        object.__setattr__(pipe, "text_encoder", te)
        try:
            with torch.no_grad():
                pe, ne = pipe.encode_prompt(prompt=p["prompt"], negative_prompt=p["negative_prompt"],
                                            do_classifier_free_guidance=True, max_sequence_length=512,
                                            device=torch.device("cpu"), dtype=torch.float32)
        finally:
            object.__setattr__(pipe, "text_encoder", None)
            del te
            gc.collect()
        cache["embeds"] = {ek: (pe.to(torch.bfloat16), ne.to(torch.bfloat16))}
    return cache["embeds"][ek]


def task_wan(torch, cache, p, progress):
    from PIL import Image

    from .worker import _save_frames

    pipe, te = _build(torch, cache, p)
    vae, d1 = cache["vae"], cache["d1"]

    pe, ne = _embeds(torch, cache, pipe, p, progress)

    if p.get("speed") == "lightning-4step" and p["variant"] != "5B":
        p = dict(p, steps=4, guidance=1.0, guidance_2=1.0)
    steps = p["steps"]

    def cb(_pipe, i, t, kw):
        progress({"step": i + 1, "steps": steps})
        return kw

    args = dict(prompt_embeds=pe.to("cuda:0"), negative_prompt_embeds=ne.to("cuda:0"),
                width=p["width"], height=p["height"], num_frames=p["frames"],
                num_inference_steps=steps, guidance_scale=p["guidance"],
                generator=torch.Generator("cpu").manual_seed(p["seed"]),
                output_type="latent", callback_on_step_end=cb)
    if getattr(pipe, "transformer_2", None) is not None:
        args["guidance_scale_2"] = p["guidance_2"]
    if p.get("image"):
        args["image"] = Image.open(p["image"]).convert("RGB").resize((p["width"], p["height"]))

    low = getattr(pipe, "transformer_2", None)
    d0 = "cuda:0"
    for attempt in range(3):
        try:
            args["generator"] = torch.Generator("cpu").manual_seed(p["seed"])
            with torch.no_grad():
                latents = pipe(**args).frames
            break
        except torch.cuda.OutOfMemoryError as e:
            # The message names the card ("GPU 0" / "GPU 1"). Stream that card's expert
            # from RAM and retry, unless it's already streaming or the user said never.
            gpu1 = "GPU 1" in str(e) and low is not None
            tag, model, dev = ("low", low.m, d1) if gpu1 else ("high", pipe.transformer, d0)
            if cache.get(f"{tag}_streamed") or p.get(f"offload_{tag}", "auto") == "never":
                raise
            progress({"text": f"{dev} out of memory -> streaming {tag}-noise expert from RAM, retrying"})
            torch.cuda.empty_cache()
            _stream(torch, cache, model, dev, tag)

    # ---- decode on the secondary card; free room there first in A14B mode ----
    progress({"text": "decoding on secondary GPU..."})
    if cache.get("low_streamed"):
        low = None  # weights already live in RAM; only one block is ever on the card
    if low is not None:
        low.m.to("cpu")
        torch.cuda.empty_cache()
    try:
        with torch.no_grad():
            lat = latents.to(d1, vae.dtype)
            z = vae.config.z_dim
            mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(lat)
            std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(lat)
            video = vae.decode(lat / std + mean, return_dict=False)[0]
            frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    finally:
        if low is not None:
            low.m.to(d1)
    return {"frames": _save_frames(frames, p["out"])}


# --------------------------------------------------------------------------- #
# Pipelined batch: both GPUs busy at the same time
# --------------------------------------------------------------------------- #
#
#   time ->      clip0          clip1          clip2
#   4070 S  [A: high steps][A: high steps][A: high steps]
#   4060 Ti                [B: low+decode ][B: low+decode ][B: low+decode]
#
# Stage A (primary card) runs the high-noise steps of clip k+1 while stage B
# (secondary card) runs the low-noise steps + VAE decode of clip k. Each clip owns a
# private copy of the scheduler; the scheduler math and latents live on the CPU
# (a few MB), so a clip can hop between cards with no device juggling.

def _denoise(torch, model, dev, sched, latents, ts, pe, ne, g, cond, cancel):
    """Run timesteps `ts` with `model` on `dev`. latents/sched stay on CPU."""
    pe_d, ne_d = pe.to(dev), ne.to(dev)
    for t in ts:
        if cancel.is_set():
            raise RuntimeError("cancelled (other stage failed)")
        x = latents.to(dev, torch.bfloat16)
        if cond is not None:
            x = torch.cat([x, cond.to(dev, torch.bfloat16)], dim=1)
        tt = t.expand(x.shape[0]).to(dev)
        with torch.no_grad():
            out = model(hidden_states=x, timestep=tt, encoder_hidden_states=pe_d, return_dict=False)[0]
            if g > 1.0:
                un = model(hidden_states=x, timestep=tt, encoder_hidden_states=ne_d, return_dict=False)[0]
                out = un + g * (out - un)
        latents = sched.step(out.float().cpu(), t, latents, return_dict=False)[0]
        yield latents


def task_wan_batch(torch, cache, p, progress):
    """Generate len(seeds) clips with the two experts working concurrently."""
    import copy
    import queue
    import threading

    from PIL import Image

    from .worker import _save_frames

    seeds = p["seeds"]
    if p["variant"] == "5B" or torch.cuda.device_count() < 2 or len(seeds) == 1:
        # nothing to overlap -> plain sequential runs
        out = []
        for i, s in enumerate(seeds):
            progress({"text": f"clip {i + 1}/{len(seeds)}"})
            out.append(task_wan(torch, cache, dict(p, seed=s, out=f"{p['out']}_{i}"), progress)["frames"])
        return {"clips": out}

    pipe, _ = _build(torch, cache, p)
    pe, ne = _embeds(torch, cache, pipe, p, progress)
    if p.get("speed") == "lightning-4step":
        p = dict(p, steps=4, guidance=1.0, guidance_2=1.0)
    steps, d0, d1 = p["steps"], "cuda:0", cache["d1"]
    high, low = pipe.transformer, pipe.transformer_2.m
    vae = cache["vae"]
    vae_lock = threading.Lock()  # Wan VAE keeps internal caches -> one user at a time

    base = copy.deepcopy(pipe.scheduler)
    base.set_timesteps(steps, device="cpu")
    boundary = pipe.config.boundary_ratio * base.config.num_train_timesteps
    ts_high = [t for t in base.timesteps if t >= boundary]
    ts_low = [t for t in base.timesteps if t < boundary]
    i2v = p["variant"] == "A14B-I2V"
    image = None
    if i2v:
        image = Image.open(p["image"]).convert("RGB").resize((p["width"], p["height"]))

    handoff = queue.Queue(maxsize=1)   # A waits if B is still busy (bounded RAM use)
    cancel = threading.Event()
    errors, results = [], [None] * len(seeds)
    done_steps = {"a": 0, "b": 0}
    total = len(seeds) * steps

    def report():
        progress({"step": done_steps["a"] + done_steps["b"], "steps": total,
                  "text": f"4070S(high) {done_steps['a']}/{len(seeds) * len(ts_high)} | "
                          f"4060Ti(low) {done_steps['b']}/{len(seeds) * len(ts_low)}"})

    def stage_a():
        torch.cuda.set_device(torch.device(d0))  # streams/offload hooks use the thread's device
        try:
            for i, seed in enumerate(seeds):
                sched = copy.deepcopy(base)
                gen = torch.Generator("cpu").manual_seed(seed)
                cond = None
                if i2v:
                    img = pipe.video_processor.preprocess(image, height=p["height"], width=p["width"]) \
                        .to(d0, torch.float32)
                    with vae_lock, torch.no_grad():
                        lat, cond = pipe.prepare_latents(
                            img, 1, vae.config.z_dim, p["height"], p["width"], p["frames"],
                            torch.float32, torch.device(d0), gen, None)
                    lat, cond = lat.cpu(), cond.cpu()
                else:
                    lat = pipe.prepare_latents(1, high.config.in_channels, p["height"], p["width"],
                                               p["frames"], torch.float32, torch.device("cpu"), gen, None)
                for lat in _denoise(torch, high, d0, sched, lat, ts_high, pe, ne,
                                    p["guidance"], cond, cancel):
                    done_steps["a"] += 1
                    report()
                handoff.put((i, sched, lat, cond))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            cancel.set()
        finally:
            handoff.put(None)

    def stage_b():
        torch.cuda.set_device(torch.device(d1))
        resident = not cache.get("low_streamed")
        try:
            while True:
                item = handoff.get()
                if item is None or cancel.is_set():
                    return
                i, sched, lat, cond = item
                for lat in _denoise(torch, low, d1, sched, lat, ts_low, pe, ne,
                                    p["guidance_2"], cond, cancel):
                    done_steps["b"] += 1
                    report()
                if resident:  # make room for the VAE on the 8 GB card
                    low.to("cpu")
                    torch.cuda.empty_cache()
                with vae_lock, torch.no_grad():
                    x = lat.to(d1, vae.dtype)
                    z = vae.config.z_dim
                    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(x)
                    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(x)
                    video = vae.decode(x / std + mean, return_dict=False)[0]
                    frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
                    del video, x
                if resident:
                    low.to(d1)
                results[i] = _save_frames(frames, f"{p['out']}_{i}")
                progress({"text": f"clip {i + 1}/{len(seeds)} finished"})
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            cancel.set()
            while handoff.get() is not None:  # unblock stage A
                pass

    ta, tb = threading.Thread(target=stage_a), threading.Thread(target=stage_b)
    ta.start(); tb.start(); ta.join(); tb.join()
    if errors:
        raise errors[0]
    return {"clips": results}
