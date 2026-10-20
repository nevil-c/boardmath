"""BoardMath: barebones live whiteboard -> LaTeX demo using Chandra via the Datalab API.

camera -> latest frame -> preprocess -> change check -> Datalab (Chandra) -> LaTeX -> web page
"""

import argparse
from collections import deque
import difflib
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
p.add_argument("--dup-similarity", type=float, default=0.85,
               help="a new line this similar (0-1) to one that just dropped out of view is the "
                    "same line misread, not a new step; >1 turns fuzzy matching off")
p.add_argument("--confirm", type=int, default=2,
               help="a new line must read the same in this many snapshots in a row before it "
                    "becomes a step (stops half-written lines); 1 = add immediately")
p.add_argument("--replace-window", type=float, default=30,
               help="seconds during which a step that vanishes and is rewritten in the same "
                    "spot gets replaced instead of kept")
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
    "key2n": {},             # every canonical key seen -> step number
    "visible": set(),        # step numbers on the board at the last snapshot
    "pending": {},           # new lines waiting for a confirming read: key -> count
    "steps_rev": 0,          # bumped whenever a step is added or completed
    "snap_count": 0,
    "skipped": 0,
    "last_diff": None,
    "pages_billed": 0,
    "est_cost": 0.0,
    "last_latency": None,
    "hand": None,            # latest normalized [x, y] of the moving hand, or None
    "hand_t": 0.0,
    "cam_fps": 0.0,
    "cam_error": None,
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


def open_cam(cam_id):
    """Camera id -> VideoCapture. Ids: "0" (index), "0@1400" (index@backend), or a file/URL."""
    if "@" in cam_id and cam_id.replace("@", "").isdigit():
        idx, backend = map(int, cam_id.split("@"))
        return cv2.VideoCapture(idx, backend)
    return cv2.VideoCapture(int(cam_id) if cam_id.isdigit() else cam_id)


def _mac_cameras():
    out = subprocess.check_output(["system_profiler", "SPCameraDataType", "-json"], timeout=5)
    cams = []
    for i, c in enumerate(json.loads(out).get("SPCameraDataType", [])):
        name = c.get("_name", f"Camera {i}")
        model = c.get("spcamera_model-id", "")
        label = f"{i}: {name}" + (f" ({model})" if "iPhone" in model and model not in name else "")
        cams.append({"id": str(i), "label": label})
    return cams


def _named_cameras():
    """Windows/Linux names. c.index encodes the backend, e.g. 1400 = MSMF device 0."""
    from cv2_enumerate_cameras import enumerate_cameras
    cams = []
    for c in enumerate_cameras():
        backend = c.backend or c.index // 100 * 100
        idx = c.index % 100
        api = cv2.videoio_registry.getBackendName(backend) if backend else "auto"
        cams.append({"id": f"{idx}@{backend}", "label": f"{c.name} ({api} {idx})"})
    return cams


def list_cameras(in_use=None):
    cams = []
    try:
        cams = _mac_cameras() if sys.platform == "darwin" else _named_cameras()
    except Exception as e:
        print("[cam] camera names unavailable:", e, flush=True)
    if not cams:
        cams = [{"id": str(i), "label": f"Camera {i}"} for i in range(4)]
    if not any(c["id"] == str(args.camera) for c in cams):
        cams.append({"id": str(args.camera), "label": f"{args.camera} (from --camera)"})
    # listed != working (e.g. OBS Virtual Camera while OBS isn't sending), so grab one frame.
    # The camera already open here is skipped: a second open of it could fail.
    for c in cams:
        if c["id"] == in_use:
            c["ok"] = True
            continue
        cap = open_cam(c["id"])
        c["ok"] = bool(cap.isOpened() and cap.read()[0])
        cap.release()
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
                  # each board is "seen" 3 times in a row, like a real still board
                  for tex, x, y in self.samples[self.i // 3 % len(self.samples)]]
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


# ---------------------------------------------------------------- duplicate checker

_SPACING = re.compile(r"\\(?:left|right|displaystyle|textstyle|quad|qquad)(?![A-Za-z])|\\[,;:! ]|~")
_SINGLE_BRACED = re.compile(r"(?<!\\)\{(\\[A-Za-z]+|[A-Za-z0-9])\}")


def canon(latex):
    """Comparison key: the same line read twice by OCR should give the same key.

    Drops whitespace, \\left/\\right, spacing commands and trailing punctuation,
    folds \\dfrac/\\tfrac to \\frac and {x} to x, so x^{2} == x^2, \\frac{a}{b} == \\frac a b.
    """
    s = _SPACING.sub("", latex)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\s+", "", s)
    prev = None
    while prev != s:
        prev, s = s, _SINGLE_BRACED.sub(r"\1", s)
    return s.rstrip(".,;")


def _same_line(k, vanished, eqs):
    """Was the new key k just a different reading of a line that dropped out?

    Only lines visible in the previous snapshot and missing now are candidates,
    so two similar lines that are both on the board stay two steps.
    Returns (step number, "grew" | "dup") or (None, None).
    """
    best, best_score = (None, None), 0.0
    for n in vanished:
        e = eqs[n - 1]["key"]
        if k.startswith(e):
            how, score = "grew", 2.0      # caught half-written last time
        elif e.startswith(k):
            how, score = "dup", 1.5       # hand covering the end of the line
        else:
            score = difflib.SequenceMatcher(None, k, e).ratio()
            how = "dup"                   # OCR jitter, e.g. m -> rn
            if score < args.dup_similarity:
                continue
        if score > best_score:
            best, best_score = (n, how), score
    return best


def _overlap(a, b):
    """Intersection over the smaller box (normalized [x0, y0, x1, y1]); 0 if either is missing."""
    if not a or not b:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def merge_steps(book, steps, when, now=None, confirm=None):
    """Fold one snapshot's lines into the step list. Returns (new, updated) items.

    book holds equations, key2n (every key seen -> step number), visible (step
    numbers in the previous snapshot) and pending (unconfirmed lines);
    merge_steps updates all four.

    A brand-new line only becomes a step once it reads the same in `confirm`
    snapshots in a row, so a line caught mid-writing (which reads differently
    each time) waits until it's finished. If a step that slipped through
    anyway disappears and a new line is confirmed in the same spot within
    --replace-window seconds, the new line replaces it.
    """
    now = time.time() if now is None else now
    confirm = args.confirm if confirm is None else confirm
    eqs, key2n = book["equations"], book["key2n"]
    lines = {}
    for b in steps:
        k = canon(b["latex"])
        if k and k not in lines:
            lines[k] = b
    present = {key2n[k] for k in lines if k in key2n}
    vanished = [n for n in sorted(book["visible"]) if n not in present]
    new, updated, visible = [], [], set(present)
    pending = {}
    for k, b in lines.items():
        if k in key2n:
            continue
        n, how = _same_line(k, vanished, eqs)
        if n is not None:
            vanished.remove(n)
            key2n[k] = n
            visible.add(n)
            if how == "grew":
                eqs[n - 1].update(latex=b["latex"], key=k, box=b.get("box"))
                updated.append(eqs[n - 1])
            continue
        count = book["pending"].get(k, 0) + 1
        if count < confirm:
            pending[k] = count   # wait for another identical read
            continue
        # a young step that's gone from the board, in the same spot = this line half-written
        old = next((e for e in reversed(eqs)
                    if not e.get("removed") and e["n"] not in present and e["n"] not in visible
                    and now - e["added"] <= args.replace_window
                    and _overlap(e.get("box"), b.get("box")) > 0.5), None)
        if old is not None:
            old.update(latex=b["latex"], key=k, box=b.get("box"))
            key2n[k] = old["n"]
            visible.add(old["n"])
            updated.append(old)
            continue
        item = {"n": len(eqs) + 1, "time": when, "latex": b["latex"], "hand": b["hand"],
                "key": k, "box": b.get("box"), "added": now}
        eqs.append(item)
        key2n[k] = item["n"]
        visible.add(item["n"])
        new.append(item)
    book["visible"] = visible
    book["pending"] = pending
    return new, updated


def remove_step(book, n):
    """Hide step n. Its readings stay in key2n, so the same line isn't re-added."""
    e = book["equations"][n - 1]
    e["removed"] = True
    book["visible"].discard(n)
    return e


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
    _selftest_dups()


def _selftest_dups():
    """The duplicate checker: re-reads merge, genuinely different lines don't."""
    book = {"equations": [], "key2n": {}, "visible": set(), "pending": {}}

    def snap(*lines):   # confirm=1: these cases are about matching, not confirmation
        new, upd = merge_steps(book, [{"latex": x, "hand": None} for x in lines], "t", confirm=1)
        return [e["latex"] for e in new], [(e["n"], e["latex"]) for e in upd]

    assert canon(r"x^{2} + \frac{a}{b}.") == canon(r"x^2+\frac a b") == canon(r"x^2 +\dfrac{a}{b}")
    assert canon(r"\left( a \right)\,b") == canon("(a)b")
    F = r"\sum_{i=1}^n F_i = ma"
    assert snap(F, "x^{2}+1") == ([F, "x^{2}+1"], [])
    assert snap(r"\sum_{i=1}^{n} F_i=ma", r"x^2 + 1.") == ([], []), "spacing/brace variants"
    assert snap(F, "x^2+1", "y = 2") == (["y = 2"], [])
    assert snap(F, "x^2+1", "y=2x+1") == ([], [(3, "y=2x+1")]), "half-written line grew"
    assert snap(r"\sum_{i=1}^n F_i = rna", "x^2+1", "y=2x+1") == ([], []), \
        "one-letter misread of a visible line"
    assert snap(F, "x^2+1", "y=2x") == ([], []), "hand covering the end of a line"
    assert snap(F, "x^2+1", "y=2x+1", "y=2x+1+z") == (["y=2x+1+z"], []), \
        "a longer line next to the short one is a new step"
    assert snap("x=2") == (["x=2"], []), "board wiped, new line"
    assert snap("x=3") == (["x=3"], []), "short lines: one changed character is a real change"
    assert [e["latex"] for e in book["equations"]] == \
        [F, "x^{2}+1", "y=2x+1", "y=2x+1+z", "x=2", "x=3"]
    _selftest_midwriting()


def _selftest_midwriting():
    """Half-written lines: wait for a confirming read; replace one that slipped through."""
    book = {"equations": [], "key2n": {}, "visible": set(), "pending": {}}
    row = lambda i: [.1, .1 * i, .5, .1 * i + .08]  # noqa: E731
    board = [("y=mx+b", 0), ("a=4", 1)]

    def snap(t, *extra):
        lines = [{"latex": x, "hand": None, "box": row(r)} for x, r in board + list(extra)]
        new, upd = merge_steps(book, lines, "t", now=t, confirm=2)
        return [e["latex"] for e in new], [(e["n"], e["latex"]) for e in upd]

    assert snap(0) == ([], []), "first read only queues the lines"
    assert snap(1) == (["y=mx+b", "a=4"], [])
    # writing C=12 on row 4: the half-drawn reads differ each time, so none is confirmed
    assert snap(2, ("a=u", 4)) == ([], [])
    assert snap(3, ("C=1", 4)) == ([], [])
    assert snap(4, ("C=12", 4)) == ([], [])
    assert snap(5, ("C=12", 4)) == (["C=12"], []), "finished line confirmed"
    board.append(("C=12", 4))
    # writer paused mid-line, so a garbage read got confirmed...
    assert snap(6, ("Z=\\sqrt{u}", 5)) == ([], [])
    assert snap(7, ("Z=\\sqrt{u}", 5)) == (["Z=\\sqrt{u}"], [])
    # ...then the finished line, unrelated text but the same spot, replaces it
    assert snap(8, ("Z=-17", 5)) == ([], [])
    assert snap(9, ("Z=-17", 5)) == ([], [(4, "Z=-17")]), "same spot replaces the garbage step"
    board.append(("Z=-17", 5))
    # outside the replace window, a rewrite in the same spot is a new step
    assert snap(50, ("q=1", 5)) == ([], [])
    board.pop()
    assert snap(51, ("q=1", 5)) == (["q=1"], [])
    # removed by hand: the same reading doesn't come back
    remove_step(book, 3)
    assert snap(52, ("q=1", 5)) == ([], [])
    assert [e["latex"] for e in book["equations"] if not e.get("removed")] == \
        ["y=mx+b", "a=4", "Z=-17", "q=1"]


if args.selftest:
    _selftest()
    print("selftest ok")
    sys.exit(0)


# ---------------------------------------------------------------- threads

def capture_loop():
    global latest_frame
    cur_cam = None
    good_cam = None  # last camera that opened, to fall back to
    cap = None
    prev_small = None
    n, t0 = 0, time.time()
    while True:
        if settings["camera"] != cur_cam:
            if cap is not None:
                cap.release()
            cur_cam = settings["camera"]
            latest_frame = None
            prev_small = None
            cap = open_cam(cur_cam)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            if not cap.isOpened():
                cap.release()
                cap = None
                label = next((c["label"] for c in CAMERAS if c["id"] == cur_cam), cur_cam)
                if good_cam is not None and good_cam != cur_cam:
                    state["cam_error"] = f"cannot open {label}, switched back"
                    settings["camera"] = good_cam  # reopened on the next pass
                else:
                    state["cam_error"] = f"cannot open {label}, pick another camera"
                print("[cam]", state["cam_error"], flush=True)
                time.sleep(0.5)
                continue
            if cur_cam != good_cam:  # a fall-back keeps its "switched back" message
                state["cam_error"] = None
            good_cam = cur_cam
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
    forced = 0  # extra reads of an unchanged board, to confirm pending lines
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
            if diff >= settings["diff_threshold"]:
                forced = 0
            elif state["pending"] and forced < args.confirm:
                forced += 1  # board looks unchanged, but a new line still needs confirming
            else:
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
        # whole board comes back each time; the checker keeps only genuinely new lines
        new, updated = merge_steps(state, steps, snap["time"])
        if new or updated:
            state["steps_rev"] += 1
        snap["steps"] = new
    print(f"[ocr] #{snap['id']} {dt:.2f}s, {len(steps)} on board, {len(new)} new, "
          f"{len(updated)} completed, {len(state['pending'])} waiting to confirm", flush=True)
    if new or updated:   # empty board writes nothing, so the transcript stays blank too
        TRANSCRIPTS.mkdir(exist_ok=True)
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(f"## Snapshot {snap['id']} @ {snap['time']} ({dt:.2f}s)\n\n"
                    + "".join(f"{e['n']}. ${e['latex']}$\n" for e in new)
                    + "".join(f"{e['n']}. (completed) ${e['latex']}$\n" for e in updated) + "\n")


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
            "steps_rev": state["steps_rev"],
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
                "cam_error": state["cam_error"],
                "transcript": str(transcript_path.relative_to(ROOT)),
            },
            "paused": state["paused"],
            "settings": settings,
            "cameras": CAMERAS,
        }


@app.post("/api/cameras/rescan")
def api_cameras_rescan():
    global CAMERAS
    CAMERAS = list_cameras(in_use=settings["camera"])
    return {"cameras": CAMERAS}


@app.post("/api/steps/remove")
async def api_remove_step(req: Request):
    n = int((await req.json())["n"])
    with lock:
        if not 1 <= n <= len(state["equations"]):
            return {"ok": False}
        e = remove_step(state, n)
        state["steps_rev"] += 1
    TRANSCRIPTS.mkdir(exist_ok=True)
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write(f"{n}. (removed) ${e['latex']}$\n\n")
    return {"ok": True}


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
