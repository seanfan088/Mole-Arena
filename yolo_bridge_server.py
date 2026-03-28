#!/opt/miniconda3/bin/python3
"""Local YOLO bridge server for Mole Arena.

Modes:
- manual HTTP posting to /api/detections
- optional synthetic detector loop for smoke testing
- optional webcam + YOLO loop (when enabled with --webcam)

The detector output is normalized into the game's bridge payload shape so the
front-end can consume it immediately.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parent
STATE: dict[str, Any] = {
    "state": "idle",
    "protocol": "http-poll",
    "detections": [],
    "updated_at": 0.0,
    "source": "manual",
}
LOCK = threading.Lock()
STOP_EVENT = threading.Event()

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mole Arena YOLO Bridge</title>
    <style>
      body {
        margin: 0;
        padding: 24px;
        font-family: sans-serif;
        background: #111;
        color: #f5f5f5;
      }
      .wrap {
        max-width: 900px;
        margin: 0 auto;
      }
      textarea {
        width: 100%;
        min-height: 180px;
        border-radius: 12px;
        border: 0;
        padding: 14px;
        font: 14px/1.5 monospace;
      }
      button {
        margin-top: 12px;
        margin-right: 10px;
        padding: 10px 16px;
        border: 0;
        border-radius: 999px;
        cursor: pointer;
      }
      pre {
        background: #1e1e1e;
        padding: 14px;
        border-radius: 12px;
        white-space: pre-wrap;
      }
      code {
        background: #1e1e1e;
        padding: 2px 6px;
        border-radius: 6px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>YOLO Bridge Server</h1>
      <p>Use this page to manually post detections or let a local detector write to <code>/api/detections</code>.</p>
      <textarea id="payload">{
  "state": "external-ready",
  "protocol": "http-poll",
  "source": "manual",
  "detections": [
    {"holeIndex": 2, "label": "mole", "confidence": 0.94},
    {"holeIndex": 5, "label": "bomb", "confidence": 0.81}
  ]
}</textarea>
      <br />
      <button id="send">POST /api/detections</button>
      <button id="refresh">GET /api/detections</button>
      <pre id="result">Waiting...</pre>
    </div>
    <script>
      const result = document.getElementById('result');
      async function refresh() {
        const response = await fetch('/api/detections');
        result.textContent = JSON.stringify(await response.json(), null, 2);
      }
      document.getElementById('send').addEventListener('click', async () => {
        try {
          const payload = JSON.parse(document.getElementById('payload').value);
          const response = await fetch('/api/detections', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          result.textContent = await response.text();
        } catch (error) {
          result.textContent = String(error);
        }
      });
      document.getElementById('refresh').addEventListener('click', refresh);
      refresh();
    </script>
  </body>
</html>
"""


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def set_state(*, state: str, protocol: str, detections: list[dict[str, Any]], source: str) -> None:
    with LOCK:
        STATE["state"] = state
        STATE["protocol"] = protocol
        STATE["detections"] = detections
        STATE["updated_at"] = time.time()
        STATE["source"] = source


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if self.path == "/api/detections":
            with LOCK:
                payload = dict(STATE)
            self._send(200, json_bytes(payload))
            return

        self._send(404, json_bytes({"error": "not_found"}))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/detections":
            self._send(404, json_bytes({"error": "not_found"}))
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send(400, json_bytes({"error": "bad_json", "detail": str(exc)}))
            return

        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            self._send(400, json_bytes({"error": "detections_must_be_list"}))
            return

        set_state(
            state=payload.get("state", "external-ready"),
            protocol=payload.get("protocol", "http-poll"),
            detections=detections,
            source=payload.get("source", "manual"),
        )
        self._send(200, json_bytes({"ok": True, "count": len(detections)}))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def synthetic_loop() -> None:
    tick = 0
    while not STOP_EVENT.is_set():
        tick += 1
        hole = tick % 9
        detections = [{"holeIndex": hole, "label": "mole", "confidence": 0.9}]
        if tick % 4 == 0:
            detections.append({"holeIndex": (hole + 3) % 9, "label": "bomb", "confidence": 0.82})
        set_state(state="external-ready", protocol="http-poll", detections=detections, source="synthetic-loop")
        STOP_EVENT.wait(0.35)


def webcam_yolo_loop(model_name: str, camera_index: int) -> None:
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_name)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        set_state(state="camera-open-failed", protocol="http-poll", detections=[], source="webcam-yolo")
        return

    set_state(state="camera-opened", protocol="http-poll", detections=[], source="webcam-yolo")

    try:
        while not STOP_EVENT.is_set():
            ok, frame = capture.read()
            if not ok:
                set_state(state="camera-read-failed", protocol="http-poll", detections=[], source="webcam-yolo")
                STOP_EVENT.wait(0.2)
                continue

            results = model.predict(frame, verbose=False, conf=0.25)
            detections: list[dict[str, Any]] = []
            if results:
                result = results[0]
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    label = str(names.get(cls_id, cls_id)).lower()
                    xyxy = box.xyxy[0].tolist()
                    center_x = (xyxy[0] + xyxy[2]) / 2.0
                    center_y = (xyxy[1] + xyxy[3]) / 2.0
                    hole_index = center_to_hole(center_x, center_y, frame.shape[1], frame.shape[0])
                    detections.append(
                        {
                            "holeIndex": hole_index,
                            "label": normalize_label(label),
                            "confidence": round(confidence, 4),
                        }
                    )

            set_state(
                state="external-ready",
                protocol="http-poll",
                detections=detections,
                source=f"webcam-yolo:{model_name}",
            )
            STOP_EVENT.wait(0.12)
    finally:
        capture.release()


def normalize_label(label: str) -> str:
    if "bomb" in label:
        return "bomb"
    return "mole"


def center_to_hole(center_x: float, center_y: float, width: int, height: int) -> int:
    col = min(2, max(0, int(center_x / max(width / 3, 1))))
    row = min(2, max(0, int(center_y / max(height / 3, 1))))
    return row * 3 + col


def start_detector_thread(args: argparse.Namespace) -> threading.Thread | None:
    if args.synthetic:
        thread = threading.Thread(target=synthetic_loop, daemon=True)
        thread.start()
        return thread

    if args.webcam:
        thread = threading.Thread(
            target=webcam_yolo_loop,
            args=(args.model, args.camera_index),
            daemon=True,
        )
        thread.start()
        return thread

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mole Arena YOLO bridge server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--synthetic", action="store_true", help="run synthetic detector loop")
    parser.add_argument("--webcam", action="store_true", help="run webcam YOLO detector")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_thread = start_detector_thread(args)
    server = ReusableThreadingHTTPServer((args.host, args.port), Handler)
    mode = "manual"
    if args.synthetic:
        mode = "synthetic-loop"
    if args.webcam:
        mode = f"webcam-yolo:{args.model}"
    print(f"YOLO bridge listening on http://{args.host}:{args.port} ({mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STOP_EVENT.set()
        if detector_thread and detector_thread.is_alive():
            detector_thread.join(timeout=1)


if __name__ == "__main__":
    main()
