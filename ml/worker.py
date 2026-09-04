"""Setu ml-worker — multi-camera orchestration.

Entry point: python -m ml.worker  (or: setu-worker)

Launches one thread per configured camera. Each thread runs an independent
reconnect loop so a single dead feed never stalls the others.

Startup sequence:
  1. Warm all AI models (so the first frame isn't slow)
  2. Apply DB schema + seed cameras (idempotent)
  3. Start per-camera threads
  4. Block on thread join (Ctrl-C to stop)
"""

import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from ml.config import settings
from ml.db import commit, get_camera_id, get_connection, write_alert, write_detection
from ml.models import warm_all_models
from ml.pipeline import DetectionPipeline
from ml.storage import upload_thumbnail
from ml.stream import CameraStream
from ml.watchlist import WatchlistChecker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_stop_event = threading.Event()


def process_camera(cam_external_id: str) -> None:
    """Run the full inference loop for one camera until _stop_event is set."""
    prefix = f"[{cam_external_id}]"
    log.info("%s Starting camera worker", prefix)

    # Each thread gets its own DB connection and pipeline instance.
    # DetectionPipeline re-uses shared model singletons (thread-safe for inference).
    conn = get_connection()
    pipeline = DetectionPipeline()
    watchlist = WatchlistChecker(conn)
    watchlist.refresh()

    camera_uuid = get_camera_id(conn, cam_external_id)
    if camera_uuid is None:
        log.error("%s Camera not found in DB — run scripts/seed.py first", prefix)
        return

    from ml.auth import get_cookie
    cookie = get_cookie()

    # ── Stopped-vehicle anomaly detector ────────────────────────────────────
    # Tracks centroid stability per track_id. If a vehicle barely moves for
    # STOPPED_FRAMES_THRESHOLD consecutive sampled frames, fire an alert.
    #
    # Tuning:
    #   STOPPED_PIXEL_THRESHOLD: max centroid drift in pixels to count as "still"
    #     (at 1080p, 20px is about 1% of frame width — parked vs slow-moving)
    #   STOPPED_FRAMES_THRESHOLD: consecutive still frames before alert fires
    #     (with frame_skip=5 at 30FPS → 1 frame per ~0.17s; 30 frames ≈ 5 seconds)
    #
    # Per track: {track_id: {"cx": float, "cy": float, "still_frames": int, "alerted": bool}}
    STOPPED_PIXEL_THRESHOLD = 20    # pixels centroid must not move per processed frame
    STOPPED_FRAMES_THRESHOLD = 30   # ~5 seconds at live 6FPS effective rate

    stopped_tracks: dict[int, dict] = {}

    frame_count = 0
    detection_count = 0

    with CameraStream(cam_external_id, cookie) as stream:
        for frame_idx, pts_ms, frame_bgr in stream.frames():
            if _stop_event.is_set():
                break

            # Refresh watchlist every ~500 processed frames (≈ a few minutes)
            if frame_idx % 500 == 0 and frame_idx > 0:
                watchlist.refresh()

            detections = pipeline.process_frame(frame_bgr, camera_uuid, pts_ms)
            frame_count += 1

            # Track which track_ids were seen this frame (to evict stale ones)
            seen_track_ids: set[int] = set()

            for det in detections:
                # Upload thumbnail to MinIO
                crop = frame_bgr[
                    det.bbox["y1"]:det.bbox["y2"],
                    det.bbox["x1"]:det.bbox["x2"],
                ]
                det.thumbnail_key = upload_thumbnail(
                    crop, det.id, cam_external_id, det.ts
                )

                # Write detection to Postgres
                write_detection(conn, det)

                # Check against watchlist → create alert if matched
                hit = watchlist.check(det)
                if hit:
                    write_alert(conn, hit, det.id)
                    log.warning(
                        "%s 🚨 WATCHLIST MATCH — entry=%s detection=%s plate=%s",
                        prefix, hit, det.id, det.plate_text,
                    )

                detection_count += 1

                # ── Stopped-vehicle detection ─────────────────────────────────
                if det.track_id is not None:
                    tid = det.track_id
                    seen_track_ids.add(tid)
                    cx = (det.bbox["x1"] + det.bbox["x2"]) / 2.0
                    cy = (det.bbox["y1"] + det.bbox["y2"]) / 2.0

                    if tid not in stopped_tracks:
                        stopped_tracks[tid] = {"cx": cx, "cy": cy, "still_frames": 0, "alerted": False}
                    else:
                        state = stopped_tracks[tid]
                        drift = ((cx - state["cx"]) ** 2 + (cy - state["cy"]) ** 2) ** 0.5
                        if drift <= STOPPED_PIXEL_THRESHOLD:
                            state["still_frames"] += 1
                        else:
                            # Vehicle moved — reset counter and centroid
                            state["still_frames"] = 0
                            state["alerted"] = False
                        # Update centroid (slow-moving vehicles drift slightly)
                        state["cx"] = cx
                        state["cy"] = cy

                        # Fire alert if threshold crossed and not already alerted
                        if state["still_frames"] >= STOPPED_FRAMES_THRESHOLD and not state["alerted"]:
                            state["alerted"] = True
                            write_alert(conn, None, det.id, alert_type="stopped_vehicle")
                            log.warning(
                                "%s 🚦 STOPPED VEHICLE — track=%d class=%s color=%s "
                                "still_frames=%d pts=%.1fs",
                                prefix, tid, det.class_name, det.dominant_color,
                                state["still_frames"], pts_ms / 1000.0,
                            )
                # ─────────────────────────────────────────────────────────────

            # Evict track states for tracks that disappeared (bbox gone for 1+ frame)
            stale = set(stopped_tracks.keys()) - seen_track_ids
            for tid in stale:
                stopped_tracks.pop(tid, None)

            # Commit every frame's detections as a batch
            if detections:
                commit(conn)

            if frame_count % 100 == 0:
                log.info(
                    "%s Processed %d frames, %d detections total",
                    prefix, frame_count, detection_count,
                )


def main() -> None:
    log.info("═" * 60)
    log.info("Setu ml-worker starting")
    log.info("Cameras: %s", settings.cameras)
    log.info("Frame skip: every %dth frame", settings.frame_skip)
    log.info("═" * 60)

    # Warm all models before spawning threads (avoid race on first load)
    log.info("Loading AI models…")
    warm_all_models()
    log.info("Models ready ✓")

    # Graceful shutdown on Ctrl-C or SIGTERM
    def _handle_signal(sig, frame):
        log.info("Received signal %s — shutting down…", sig)
        _stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cam_ids = settings.camera_ids()
    log.info("Starting %d camera workers: %s", len(cam_ids), cam_ids)

    threads = []
    for cam_id in cam_ids:
        t = threading.Thread(
            target=process_camera,
            args=(cam_id,),
            name=f"cam-{cam_id}",
            daemon=True,
        )
        threads.append(t)
        t.start()
        time.sleep(0.5)  # stagger starts to avoid GPU init race

    log.info("All camera workers started. Press Ctrl-C to stop.")

    # Wait for stop signal
    _stop_event.wait()

    log.info("Waiting for camera threads to finish…")
    for t in threads:
        t.join(timeout=10)

    log.info("Setu ml-worker stopped.")


if __name__ == "__main__":
    main()
