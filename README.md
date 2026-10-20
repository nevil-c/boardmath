# BoardMath (barebones demo), Datalab API branch

Live handwritten whiteboard math → LaTeX. This branch runs [Chandra OCR](https://github.com/datalab-to/chandra) through the **hosted Datalab API**, so nothing runs locally and there's no model download or GPU needed. The `main` branch runs Chandra locally instead.

```
camera (OpenCV) ─► latest frame ─► preprocess ─► changed? ─► Datalab /convert (Chandra) ─► LaTeX ─► web page + transcript
                                   gray, contrast/     (skip if the        submit, then poll
                                   brightness, CLAHE,   board hasn't        request_check_url
                                   B&W threshold,       changed)
                                   resize
```

- One API request is in flight at a time. `--interval` (default 2.5 s) keeps you under the free tier's 25 requests/min.
- Every request OCRs the whole board. Lines already on the list are subtracted, so a re-read of the same board adds nothing and the steps never repeat.
- Frames whose 160 px thumbnail differs from the last OCR'd frame by less than `--diff-threshold` are skipped, so an unchanged board isn't billed.
- The page shows the live feed, the exact preprocessed frame that was uploaded, and the steps so far rendered with KaTeX (the newest is highlighted). KaTeX also emits MathML, which screen readers can read. The page also shows pages billed and an estimated cost.
- An empty board produces no step: nothing is rendered, no description of the photograph.
- Every new step is appended to `transcripts/lecture_<timestamp>.md` with its step number.

## Cost

Datalab bills each uploaded frame as one page:

| `--mode` | Price | Notes |
|---|---|---|
| `fast` | $4 / 1,000 pages | lowest latency |
| `balanced` (default) | $4 / 1,000 pages | recommended |
| `accurate` | $10 / 1,000 pages | try it if balanced misreads handwriting |

A 75-minute lecture is roughly 450–1,500 frames, so about $2–6 in balanced mode. The free plan includes $10/month on a personal email ($20 on a work email). These are list prices from datalab.to/pricing as of October 2026.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATALAB_API_KEY=...        # from datalab.to; or put DATALAB_API_KEY=... in local.env
```

macOS asks for camera permission for your terminal app on the first run. Allow it under System Settings → Privacy & Security → Camera, then rerun.

## Run

Check the API on one photo first:

```bash
python app.py --once board_photo.jpg
```

This prints the markdown and the extracted equations, and writes `once_processed.png` so you can see exactly what was uploaded.

Then start the live demo:

```bash
python app.py --camera 0
# open http://127.0.0.1:8000
```

`--backend mock` returns canned LaTeX without calling the API, which is free and good for testing the camera and UI.

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `balanced` | `fast` / `balanced` / `accurate` |
| `--camera` | `0` | camera index, or a video file / stream URL (video files loop) |
| `--interval` | `2.5` | minimum seconds between API calls |
| `--diff-threshold` | `4.0` | lower = re-OCR on smaller changes (more pages billed) |
| `--alpha` / `--beta` | `1.3` / `10` | contrast gain / brightness offset |
| `--no-bw` | off | skip the adaptive B&W threshold (sometimes better for the model) |
| `--invert` | off | for chalk on blackboard |
| `--max-side` | `1600` | resize long edge before upload |
| `--timeout` | `60` | give up on one request after N seconds |

B&W, invert, interval and diff threshold can also be changed live from the page.

## Files

- `app.py`: capture thread, OCR worker thread, preprocessing, Datalab client, FastAPI server
- `static/index.html`: the page (polls `/api/state` every second)

## Not in this demo

Pan-tilt tracking, ReSpeaker audio timestamps, braille (Nemeth/UEB) output, and 5G/MEC deployment.
