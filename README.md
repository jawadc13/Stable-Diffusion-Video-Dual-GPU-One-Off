# DualGPU Video Studio

A ComfyUI-style node editor for AI video that uses **both** of your GPUs
(RTX 4070 Super 12 GB + RTX 4060 Ti) at the same time.

- **MiniMax (Hailuo) video**: text/image to video through the MiniMax cloud API. The clip is downloaded as frames so your GPUs can keep working on it.
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

For MiniMax, get an API key from the MiniMax platform. Set `MINIMAX_API_KEY` or paste the key into the node.
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
- The default workflow runs this chain: prompt, MiniMax clip, last frame, SVD continuation (dual-GPU pipeline), concat, 2x interpolation on both GPUs, then save as MP4.

## Notes for your hardware
- PCIe: on Z170 the two cards usually run x8/x8. This is fine, because only images and latents cross the bus.
- System RAM: 16 GB or more is recommended. CPU offload on the 8 GB card uses system RAM.
- Outputs go to `output/`, uploads to `input/`, and intermediates to `temp/` (safe to delete).

## Adding nodes
Subclass `Node` in `app/nodes.py` and decorate it with `@node`. Declare `INPUTS`, `WIDGETS` and `OUTPUTS`, then write `run()`.
Heavy GPU work goes into a task function in `app/worker.py`, which you call with `ctx.pool.submit(kind, params, gpu)`.
