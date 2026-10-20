# BoardMath (barebones demo)

Live handwritten whiteboard math → LaTeX, using [Chandra OCR](https://github.com/datalab-to/chandra).

```
camera (OpenCV) ─► latest frame ─► preprocess ─► changed? ─► Chandra ─► LaTeX ─► web page + transcript
                                   gray, contrast/     (skip if the
                                   brightness, CLAHE,   board hasn't
                                   B&W threshold,       changed)
                                   resize
```

- The OCR runs back-to-back with one request in flight, so frames are processed as fast as the model allows. Use `--interval` to set a minimum gap between requests.
- Frames whose 160 px thumbnail differs from the last OCR'd frame by less than `--diff-threshold` are skipped.
- The page shows the live feed, the exact preprocessed frame sent to OCR, the raw LaTeX/markdown, and that output rendered with KaTeX. KaTeX also emits MathML, which screen readers can read.
- Every unique equation goes into a clickable lecture history, so you can navigate back.
- Every snapshot is appended to `transcripts/lecture_<timestamp>.md`.

## Setup (Apple silicon Mac, e.g. M5 / 24 GB)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first `hf` run downloads `datalab-to/chandra-ocr-2`, which is about 10 GB at BF16. The weights are not part of this repo, and they are under a modified OpenRAIL-M license (see the Chandra repo).

macOS asks for camera permission for your terminal app on the first run. Allow it under System Settings → Privacy & Security → Camera, then rerun.

## Run

Check the model on one photo first:

```bash
python app.py --once board_photo.jpg --backend hf
```

This prints the markdown and the extracted equations, and writes `once_processed.png` so you can see exactly what the model saw.

Then start the live demo:

```bash
python app.py --backend hf --camera 0
# open http://127.0.0.1:8000
```

### Backends

| `--backend` | What it does | Speed |
|---|---|---|
| `hf` (default) | Chandra via transformers on MPS (`TORCH_DEVICE=mps` set automatically) | slowest; zero extra setup |
| `server` | Chandra's `vllm` client pointed at any OpenAI-compatible server | fastest on a Mac with an MLX quant |
| `mock` | canned LaTeX, ~1 s fake latency | for testing camera/UI without the model |

**Faster option: an MLX quant served by LM Studio**

1. In LM Studio, download an MLX build of the model, e.g. `mlx-community/chandra-ocr-2-oQ8`, or a 4-bit or 8-bit variant such as `1qh/chandra-ocr-2-4bit-mlx`.
2. Load the model and start the local server (Developer tab). It listens on `http://localhost:1234/v1` by default.
3. Run:
   ```bash
   python app.py --backend server --api-base http://localhost:1234/v1 --model-name <model id shown in LM Studio>
   ```
   `mlx_vlm.server` or a real vLLM box (the "MEC" in the full design) works the same way. Point `--api-base` at it.

Chandra's own prompt (`ocr_layout`) and output parsers are used for every backend.

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--camera` | `0` | camera index, or a video file / stream URL (video files loop) |
| `--interval` | `0` | minimum seconds between OCR calls |
| `--diff-threshold` | `4.0` | lower = re-OCR on smaller changes |
| `--alpha` / `--beta` | `1.3` / `10` | contrast gain / brightness offset |
| `--no-bw` | off | skip the adaptive B&W threshold (sometimes better for the model) |
| `--invert` | off | for chalk on blackboard |
| `--max-side` | `1600` | resize long edge before OCR |
| `--max-tokens` | `2048` | max output tokens per frame |

B&W, invert, interval and diff threshold can also be changed live from the page. The camera picker on the page lists cameras by name, greys out any that give no frames (e.g. OBS Virtual Camera while OBS isn't sending), and also accepts a camera index, video file or stream URL.

## Files

- `app.py`: capture thread, OCR worker thread, preprocessing, Chandra backends, FastAPI server
- `static/index.html`: the page (polls `/api/state` every second)

## Not in this demo

Pan-tilt tracking, ReSpeaker audio timestamps, braille (Nemeth/UEB) output, and 5G/MEC deployment.
