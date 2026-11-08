# DualGPU Video Studio

A ComfyUI-style node editor for AI video that uses **both** of your GPUs
(RTX 4070 Super 12 GB + RTX 4060 Ti) at the same time.

- **Wan 2.2 (default, no API key)**: Alibaba's open-weights video model (Apache 2.0) running 100% locally, split across both GPUs. It's the closest open model to MiniMax Hailuo.
- **MiniMax (Hailuo) video (optional)**: cloud API, needs a paid key.
- **Stable Video Diffusion (local)**: image to video on your own cards, with three dual-GPU modes.
- **Frame tools**: upscale and interpolate, split across both GPUs.

## How the two GPUs are used

Each GPU gets its own worker **process** (`CUDA_VISIBLE_DEVICES` is set before torch is imported),
so each card has its own CUDA context and model cache, and they run in parallel.
That matters on a 4-core i5-6600K: the Python GIL can't serialize the cards.

| `SVDDualGPU` mode | What happens |
|---|---|
| `parallel` | Clips (`batch` seeds) alternate between the cards. Both render full clips at once. Best throughput. |
| `pipeline` | The 4070 Super does the UNet denoising and returns latents. The 4060 Ti VAE-decodes them while the 4070 S starts the next clip. Saves VRAM on the big card, so you can use higher res or more frames. |
| `single` | Only `primary_gpu`. |

| `WanVideo` model | Primary GPU (4070 Super) | Secondary GPU (4060 Ti) | CPU / RAM |
|---|---|---|---|
| `A14B` (best) | high-noise expert: layout and motion | low-noise expert: detail, plus VAE decode | UMT5 text encoder, fp32 (~22 GB) |
| `TI2V-5B` (faster) | transformer | VAE encode and decode | UMT5 text encoder |

Wan 2.2 A14B is a two-expert model: the high-noise expert does the early steps and the low-noise expert does the late ones.
Each expert gets its own card, and only a small latent tensor crosses PCIe per step. The experts are GGUF-quantized
(QuantStack repos, downloaded automatically on first run):
- 4060 Ti **8 GB**: keep `quant_low` at `Q3_K_M`.
- 4060 Ti **16 GB**: raise `quant_low` to `Q5_K_M` or `Q8_0` for better quality.
- `quant_high` `Q4_K_M` fits the 12 GB 4070 Super at 832x480. Use 720p with `Q3_K_M`, or fewer frames.
- Prompt embeddings are cached, so re-running with a new seed skips the text encoder.
- Connect an image to `WanVideo` to get image-to-video.
- Expect several minutes per 5 s clip at 480p for A14B. TI2V-5B is much faster.

The per-card workers and the dual worker never hold models at the same time.
Switching between Wan and the per-card nodes (SVD, upscale, interpolate) unloads the other group automatically.

`UpscaleDualGPU` and `InterpolateDualGPU` split the frame list in half, one half per GPU.
GPUs are sorted by VRAM, so the 4070 Super is the primary card. An 8 GB 4060 Ti automatically gets
`enable_model_cpu_offload()` (`low_vram_8gb: auto`).

## Install (Windows or Linux)

```bash
python -m venv venv
venv\Scripts\activate            # Linux: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

SVD weights download from Hugging Face on first use (~10 GB). The `img2vid-xt` models are gated,
so accept the license on HF and run `huggingface-cli login` first.

Wan 2.2 needs no account or key. The first run downloads about 20 GB (GGUF experts + text encoder + VAE) into the Hugging Face cache.
The default workflow is: prompt, Wan 2.2 A14B, 2x interpolation on both GPUs, then a 32 fps MP4.

For MiniMax (optional), get an API key from the MiniMax platform. Set `MINIMAX_API_KEY` or paste the key into the node.
If your account is on the China endpoint, change `api_base` to `https://api.minimaxi.com`.

## Run

```bash
run.bat          # or ./run.sh
```
Open http://127.0.0.1:8188. Use `--gpus 0,1` to choose cards.

## Using the editor
- Double-click or right-click the canvas to add a node. Drag from an output dot to an input dot to connect them (types must match by color).
- Click an input dot to disconnect it. Press Del to delete the selected node. Press Ctrl+Enter to queue.
- Save/Load workflows as JSON. The graph is also autosaved in your browser.

## Notes for your hardware
- PCIe: on Z170 the two cards usually run x8/x8. This is fine, because only images and latents cross the bus.
- System RAM: 48 GB is plenty. The Wan text encoder takes about 22 GB, and the rest is headroom for model loading.
- Outputs go to `output/`, uploads to `input/`, and intermediates to `temp/` (safe to delete).

## Adding nodes
Subclass `Node` in `app/nodes.py` and decorate it with `@node`. Declare `INPUTS`, `WIDGETS` and `OUTPUTS`, then write `run()`.
Heavy GPU work goes into a task function in `app/worker.py`, which you call with `ctx.pool.submit(kind, params, gpu)`.
