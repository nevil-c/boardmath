"""BoardMath: barebones live whiteboard -> LaTeX demo using Chandra via the Datalab API.

camera -> latest frame -> preprocess -> change check -> Datalab (Chandra) -> LaTeX -> web page
"""

import argparse
from collections import deque
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).parent
DATALAB_URL = "https://www.datalab.to/api/v1/convert"

# ---------------------------------------------------------------- args

p = argparse.ArgumentParser(description="BoardMath live demo")
p.add_argument("--backend", choices=["datalab", "mock"], default="datalab",
               help="datalab = hosted Chandra via the Datalab API (needs DATALAB_API_KEY); "
                    "mock = canned output")
p.add_argument("--mode", choices=["fast", "balanced", "accurate"], default="balanced",
               help="Datalab mode: fast/balanced = $4 per 1k pages, accurate = $10 per 1k pages")
p.add_argument("--camera", default="0", help="camera index or video file/URL")
p.add_argument("--interval", type=float, default=2.5,
               help="min seconds between OCR calls (default 2.5 keeps under the free tier's "
                    "25 requests/min; 0 = back-to-back)")
p.add_argument("--diff-threshold", type=float, default=4.0,
               help="mean abs pixel diff (0-255) needed to re-OCR; lower = more sensitive")
p.add_argument("--alpha", type=float, default=1.3, help="contrast gain")
p.add_argument("--beta", type=float, default=10, help="brightness offset")
p.add_argument("--no-bw", dest="bw", action="store_false", help="skip adaptive threshold to B&W")
p.add_argument("--invert", action="store_true", help="invert (chalk on blackboard)")
p.add_argument("--max-side", type=int, default=1600, help="resize long side before OCR")
p.add_argument("--timeout", type=float, default=60, help="give up on one API request after N s")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--once", metavar="IMAGE", help="OCR a single image, print result, exit")
p.add_argument("--selftest", action="store_true", help="check step ordering and exit")
p.add_argument("--hand", action="store_true",
               help="track the writing hand and order steps by it (off = reading order)")
args = p.parse_args()

# ---------------------------------------------------------------- shared state

lock = threading.Lock()
state = {
    "raw_jpg": None,         # latest camera frame as jpeg bytes
    "proc_jpg": None,        # latest preprocessed frame as jpeg bytes
    "latest": None,          # latest snapshot dict
    "equations": [],         # ordered unique equations across the lecture
    "seen": set(),
    "snap_count": 0,
    "skipped": 0,
    "last_diff": None,
    "pages_billed": 0,
    "est_cost": 0.0,
    "last_latency": None,
    "hand": None,            # latest normalized [x, y] of the moving hand, or None
    "hand_t": 0.0,
    "cam_fps": 0.0,
    "status": "starting",
    "paused": False,
}
settings = {
    "camera": str(args.camera),
    "bw": args.bw,
    "invert": args.invert,
    "interval": args.interval,
    "diff_threshold": args.diff_threshold,
}
latest_frame = None  # raw BGR ndarray, written by capture thread


def list_cameras():
    cams = []
    try:
        out = subprocess.check_output(["system_profiler", "SPCameraDataType", "-json"], timeout=5)
        for i, c in enumerate(json.loads(out).get("SPCameraDataType", [])):
            name = c.get("_name", f"Camera {i}")
            model = c.get("spcamera_model-id", "")
            label = f"{i}: {name}" + (f" ({model})" if "iPhone" in model and model not in name else "")
            cams.append({"id": str(i), "label": label})
    except Exception:
        cams = [{"id": str(i), "label": f"Camera {i}"} for i in range(4)]
    if not any(c["id"] == str(args.camera) for c in cams):
        cams.append({"id": str(args.camera), "label": str(args.camera)})
    return cams


CAMERAS = list_cameras()

TRANSCRIPTS = ROOT / "transcripts"
transcript_path = TRANSCRIPTS / f"lecture_{datetime.now():%Y%m%d_%H%M%S}.md"

# ---------------------------------------------------------------- preprocessing

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = args.max_side / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    g = cv2.convertScaleAbs(g, alpha=args.alpha, beta=args.beta)
    g = clahe.apply(g)
    g = cv2.medianBlur(g, 3)
    if settings["invert"]:
        g = 255 - g
    th = thumb(g)  # change detection on gray, before thresholding amplifies sensor noise
    if settings["bw"]:
        # dark ink on light board -> black on white
        g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    return g, th


def thumb(gray):
    t = cv2.resize(gray, (160, int(160 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(t, (5, 5), 0).astype(np.int16)


# ---------------------------------------------------------------- hand tracking

HAND_HIST = deque(maxlen=3600)   # (t, x, y) normalized, oldest first


def find_hand(prev, cur):
    """Largest blob that moved between two small grayscale frames -> normalized (x, y).

    On a board the only thing that moves that size is the writing hand. ponytail:
    no hand model, just motion — swap in mediapipe if the arm/person keeps winning.
    """
    d = cv2.absdiff(prev, cur)
    d = cv2.threshold(cv2.GaussianBlur(d, (5, 5), 0), 18, 255, cv2.THRESH_BINARY)[1]
    d = cv2.dilate(d, np.ones((7, 7), np.uint8), iterations=1)
    n, _, stats, cents = cv2.connectedComponentsWithStats(d)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if not 0.0008 < stats[i, cv2.CC_STAT_AREA] / d.size < 0.2:
        return None   # nothing moved, or the whole room moved
    h, w = prev.shape
    return [round(float(cents[i][0]) / w, 4), round(float(cents[i][1]) / h, 4)]


# ---------------------------------------------------------------- OCR backends

class MockOCR:
    samples = [
        [("F = ma", .06, .10), ("a = \\frac{dv}{dt}", .06, .30)],
        [("\\int_0^1 x^2\\,dx = \\frac{1}{3}", .05, .20)],
        [("e^{i\\pi} + 1 = 0", .10, .45)],
        [("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}", .04, .60)],
    ]

    def __init__(self):
        self.i = 0

    def __call__(self, pil):
        time.sleep(1.0)  # pretend inference
        blocks = [{"type": "equation", "latex": tex, "text": tex,
                   "box": [x, y, x + 0.55, y + 0.10], "t": 0.0, "hand": None}
                  for tex, x, y in self.samples[self.i % len(self.samples)]]
        self.i += 1
        return blocks


class DatalabOCR:
    """Submit to /convert, then poll request_check_url until complete."""

    PRICE_PER_PAGE = {"fast": 0.004, "balanced": 0.004, "accurate": 0.010}

    def __init__(self):
        key = os.environ.get("DATALAB_API_KEY")
        if not key and (ROOT / "local.env").exists():
            for line in (ROOT / "local.env").read_text().splitlines():
                if line.startswith("DATALAB_API_KEY="):
                    key = line.split("=", 1)[1].strip()
        if not key:
            sys.exit("set DATALAB_API_KEY (env var or local.env) or use --backend mock")
        self.s = requests.Session()
        self.s.headers["X-API-Key"] = key

    def __call__(self, pil):
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        r = self.s.post(
            DATALAB_URL,
            files={"file": ("board.png", buf.getvalue(), "image/png")},
            data={"output_format": "json", "mode": args.mode,
                  "disable_image_extraction": "true"},
            timeout=30,
        )
        r.raise_for_status()
        sub = r.json()
        if sub.get("success") is False:
            raise RuntimeError(sub.get("error") or "submit failed")
        check_url = sub["request_check_url"]
        deadline = time.time() + args.timeout
        while True:
            time.sleep(0.5)
            res = self.s.get(check_url, timeout=30)
            res.raise_for_status()
            res = res.json()
            if res.get("status") == "complete":
                break
            if time.time() > deadline:
                raise RuntimeError(f"timed out after {args.timeout}s")
        if not res.get("success", True):
            raise RuntimeError(res.get("error") or "conversion failed")
        if res.get("json") is None and res.get("result_url"):
            # result_url is self-authorizing: don't send the API key there
            res = requests.get(res["result_url"], timeout=30).json()
        pages = res.get("page_count") or 1
        with lock:
            state["pages_billed"] += pages
            state["est_cost"] += pages * self.PRICE_PER_PAGE[args.mode]
        return blocks_from_json(res.get("json"), (pil.width, pil.height))


def norm(e):
    return re.sub(r"\s+", "", e)


BLOCK_TYPES = {"equation", "text", "sectionheader", "listitem"}


def blocks_from_json(page, size):
    """Datalab's json page tree -> flat blocks, LaTeX only where there is math.

    Prose blocks (image captions like "A horizontal blackboard with a chalk
    sketch of a horse") keep their text but get no latex, so they never become
    a step and an empty board renders blank.
    """
    w, h = size
    out, stack = [], list((page or {}).get("children") or [])
    while stack:
        b = stack.pop()
        stack.extend(b.get("children") or [])
        kind = (b.get("block_type") or "").lower()
        if kind not in BLOCK_TYPES:
            continue
        soup = BeautifulSoup(b.get("html") or "", "html.parser")
        latex = " ".join(" ".join(m.get_text(" ").split()) for m in soup.find_all("math"))
        x0, y0, x1, y1 = (b.get("bbox") or [0, 0, w, h])[:4]
        out.append({"type": kind, "latex": latex.strip(),
                    "text": " ".join(soup.get_text(" ").split()),
                    "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "t": 0.0, "hand": None})
    return out


def order_steps(blocks, start, now):
    """Order this snapshot's equations by when the hand wrote them.

    start..now is the window since the previous OCR, so the hand crossing the
    board earlier in the lecture can't reorder an equation that was already up.
    No hand visit -> fall back to reading order, after the written ones.
    """
    def wrote_at(box):
        x0, y0, x1, y1 = box
        for t, x, y in HAND_HIST:
            if t > start and x0 - .01 <= x <= x1 + .01 and y0 - .01 <= y <= y1 + .01:
                return t, [x, y]
        return now + y0, None   # reading order: lower rows sort later

    steps = []
    for b in blocks:
        if not b.get("latex"):
            continue
        b["t"], b["hand"] = wrote_at(b["box"])
        steps.append(b)
    steps.sort(key=lambda b: b["t"])
    return steps


def run_ocr(ocr, gray):
    pil = Image.fromarray(gray).convert("RGB")
    t0 = time.time()
    blocks = ocr(pil)
    return blocks, time.time() - t0


def _selftest():
    """Hand order beats OCR order; a visit before the last snapshot can't reorder."""
    HAND_HIST.clear()
    HAND_HIST.extend([(10.0, .1, .1), (11.0, .5, .5)])
    blocks = [{"latex": "b", "box": [.45, .45, .6, .6]},
              {"latex": "a", "box": [.05, .05, .2, .2]},
              {"latex": "c", "box": [.05, .7, .2, .8]}]
    assert [b["latex"] for b in order_steps(blocks, 0, 12)] == ["a", "b", "c"]
    assert [b["latex"] for b in order_steps(blocks, 10.5, 12)] == ["b", "a", "c"]
    caption = {"block_type": "Text", "bbox": [0, 0, 10, 10],
               "html": "<p>A horizontal blackboard with a chalk sketch of a horse.</p>"}
    got = blocks_from_json({"children": [caption]}, (10, 10))
    assert got[0]["latex"] == "" and "horse" in got[0]["text"], got
    assert order_steps(got, 0, 12) == [], "a caption must not become a step"


if args.selftest:
    _selftest()
    print("selftest ok")
    sys.exit(0)


# ---------------------------------------------------------------- threads

def capture_loop():
    global latest_frame
    cur_cam = None
    cap = None
    prev_small = None
    n, t0 = 0, time.time()
    while True:
        if settings["camera"] != cur_cam:
            if cap is not None:
                cap.release()
            cur_cam = settings["camera"]
            latest_frame = None
            src = int(cur_cam) if cur_cam.isdigit() else cur_cam
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            if not cap.isOpened():
                state["status"] = f"cannot open camera {cur_cam}"
                print("[cam]", state["status"], flush=True)
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            print(f"[cam] #{cur_cam} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                  flush=True)
            n, t0 = 0, time.time()
        if cap is None:
            time.sleep(0.2)
            continue
        ok, frame = cap.read()
        if not ok:
            if not cur_cam.isdigit():  # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue
        latest_frame = frame
        if args.hand:
            g_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (240, 160))
            if prev_small is not None:
                pos = find_hand(prev_small, g_small)
                if pos:
                    HAND_HIST.append((time.time(), *pos))
                    with lock:
                        state["hand"], state["hand_t"] = pos, time.time()
            prev_small = g_small
        n += 1
        if n % 3 == 0:  # preview jpeg at reduced rate/size
            small = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
            with lock:
                state["raw_jpg"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if time.time() - t0 >= 1:
            state["cam_fps"] = round(n / (time.time() - t0), 1)
            n, t0 = 0, time.time()


def ocr_loop(ocr):
    last_thumb = None
    last_sent = 0.0
    prev_shot = 0.0
    state["status"] = "running"
    while True:
        if state["paused"] or latest_frame is None:
            time.sleep(0.1)
            continue
        wait = settings["interval"] - (time.time() - last_sent)
        if wait > 0:
            time.sleep(min(wait, 0.1))
            continue
        shot_t = time.time()
        gray, th = preprocess(latest_frame)
        with lock:
            state["proc_jpg"] = cv2.imencode(".jpg", gray)[1].tobytes()
        if last_thumb is not None and th.shape == last_thumb.shape:
            diff = float(np.abs(th - last_thumb).mean())
            state["last_diff"] = round(diff, 2)
            if diff < settings["diff_threshold"]:
                state["skipped"] += 1
                time.sleep(0.2)
                continue
        last_sent = shot_t
        state["status"] = "ocr..."
        try:
            blocks, dt = run_ocr(ocr, gray)
            steps = order_steps(blocks, prev_shot, shot_t)
        except Exception as e:
            state["status"] = f"ocr error: {e}"
            print("[ocr]", state["status"], flush=True)
            time.sleep(1)
            continue
        last_thumb = th
        add_snapshot(steps, dt, shot_t)
        prev_shot = shot_t
        state["status"] = "running"


def add_snapshot(steps, dt, shot_t):
    with lock:
        state["snap_count"] += 1
        snap = {
            "id": state["snap_count"],
            "time": datetime.now().strftime("%H:%M:%S"),
            "latency": round(dt, 2),
            "steps": [{"latex": b["latex"], "hand": b["hand"]} for b in steps],
        }
        state["latest"] = snap
        state["last_latency"] = snap["latency"]
        new = []
        for b in steps:              # whole board comes back; only unseen lines are steps
            k = norm(b["latex"])
            if k not in state["seen"]:
                state["seen"].add(k)
                item = {"n": len(state["equations"]) + 1, "time": snap["time"],
                        "latex": b["latex"], "hand": b["hand"]}
                state["equations"].append(item)
                new.append(item)
        snap["steps"] = new
    print(f"[ocr] #{snap['id']} {dt:.2f}s, {len(steps)} on board, {len(new)} new", flush=True)
    if new:   # empty board writes nothing, so the transcript stays blank too
        TRANSCRIPTS.mkdir(exist_ok=True)
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(f"## Snapshot {snap['id']} @ {snap['time']} ({dt:.2f}s)\n\n"
                    + "\n".join(f"{e['n']}. ${e['latex']}$" for e in new) + "\n\n")


# ---------------------------------------------------------------- one-shot mode

def make_ocr():
    return MockOCR() if args.backend == "mock" else DatalabOCR()


if args.once:
    img = cv2.imread(args.once)
    if img is None:
        sys.exit(f"cannot read {args.once}")
    ocr = make_ocr()
    gray, _ = preprocess(img)
    cv2.imwrite(str(ROOT / "once_processed.png"), gray)
    blocks, dt = run_ocr(ocr, gray)
    steps = order_steps(blocks, 0, time.time())
    print(f"--- {len(steps)} steps ({dt:.2f}s) ---")
    for i, b in enumerate(steps, 1):
        print(f"{i}. {b['latex']}")
    if not steps:
        print("(blank board: no math detected, nothing to show)")
    sys.exit(0)

# ---------------------------------------------------------------- web server

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, Response, StreamingResponse  # noqa: E402

app = FastAPI()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/video")
def video():
    def gen():
        while True:
            with lock:
                jpg = state["raw_jpg"]
            if jpg:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/processed.jpg")
def processed():
    with lock:
        jpg = state["proc_jpg"]
    return Response(jpg or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def api_state():
    with lock:
        return {
            "latest": state["latest"],
            "equations": state["equations"],
            "hand": state["hand"] if time.time() - state["hand_t"] < 2 else None,
            "stats": {
                "status": state["status"],
                "backend": args.backend + ("/" + args.mode if args.backend == "datalab" else ""),
                "pages_billed": state["pages_billed"],
                "est_cost": round(state["est_cost"], 3),
                "snapshots": state["snap_count"],
                "skipped": state["skipped"],
                "last_diff": state["last_diff"],
                "last_latency": state["last_latency"],
                "cam_fps": state["cam_fps"],
                "transcript": str(transcript_path.relative_to(ROOT)),
            },
            "paused": state["paused"],
            "settings": settings,
            "cameras": CAMERAS,
        }


@app.post("/api/pause")
def api_pause():
    state["paused"] = not state["paused"]
    return {"paused": state["paused"]}


@app.post("/api/settings")
async def api_settings(req: Request):
    body = await req.json()
    if "camera" in body:
        settings["camera"] = str(body["camera"])
    for k in ("bw", "invert"):
        if k in body:
            settings[k] = bool(body[k])
    for k in ("interval", "diff_threshold"):
        if k in body:
            settings[k] = float(body[k])
    return settings


if __name__ == "__main__":
    import uvicorn
    ocr = make_ocr()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=ocr_loop, args=(ocr,), daemon=True).start()
    print(f"open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
