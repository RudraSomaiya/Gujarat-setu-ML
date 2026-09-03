"""
ml.stream
=========
Camera stream abstraction for the Setu ML worker.

Connection strategy (per camera):
  1. If port 8554 is TCP-reachable → use authenticated RTSP (fast, low-latency)
  2. Otherwise → use HLS with manual AES-128 decryption (no grey smear, works anywhere)

The HLS path manually fetches encrypted .ts segments, decrypts them with
pycryptodome, writes each segment to a temp file, and decodes frames with
OpenCV.  Because every HLS segment starts with an IDR keyframe, this path
is completely immune to the "grey smear" caused by missing reference frames
on RTSP joins and reconnects.

A background prefetch thread downloads the next segment while the current
one is being decoded, hiding the network latency between segments.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
import threading
import time
from typing import Generator, Optional

import cv2
import requests
from Crypto.Cipher import AES

from ml.config import settings

log = logging.getLogger(__name__)

# User-Agent required by the Sentinel gateway for stream/segment endpoints.
# Without a Mozilla/-prefixed UA, the server returns 403 "browser required".
_MOZILLA_UA = (
    "Mozilla/5.0 (compatible; SetuML/1.0; Gujarat Police Innovation Challenge)"
)


# ─── RTSP reachability probe ──────────────────────────────────────────────────

def is_rtsp_reachable(cam_id: str) -> bool:  # noqa: ARG001 — cam_id unused but kept for API symmetry
    """Return True if the RTSP gateway TCP port is reachable from here."""
    try:
        s = socket.create_connection(("103.250.160.189", settings.rtsp_port), timeout=2)
        s.close()
        return True
    except OSError:
        return False


# ─── HLS segment fetcher ──────────────────────────────────────────────────────

class _HLSFetcher:
    """Handles HLS manifest parsing, AES key caching and segment download/decrypt."""

    def __init__(self, cookie: str) -> None:
        self._session = requests.Session()
        self._session.cookies.set("sentinel", cookie)
        self._session.headers.update({"User-Agent": _MOZILLA_UA})
        self._aes_key: Optional[bytes] = None
        self._key_lock = threading.Lock()

    # ── manifest ──────────────────────────────────────────────────────────────

    def fetch_manifest(self, cam_id: str) -> tuple[list[str], str]:
        """Fetch index.m3u8 and return (segment_urls, key_url)."""
        url = settings.hls_url(cam_id)
        r = self._session.get(url, timeout=20)
        r.raise_for_status()

        key_url: Optional[str] = None
        segments: list[str] = []
        base = f"{settings.sentinel_base_url}/{cam_id}"

        for raw_line in r.text.splitlines():
            line = raw_line.strip()
            if line.startswith("#EXT-X-KEY"):
                for part in line.split(","):
                    if part.startswith("URI="):
                        key_url = part[4:].strip('"')
            elif line and not line.startswith("#"):
                # Segment file — may be relative
                if line.startswith("http"):
                    segments.append(line)
                else:
                    segments.append(f"{base}/{line}")

        if key_url is None:
            raise RuntimeError(f"No #EXT-X-KEY found in manifest for {cam_id}")

        return segments, key_url

    # ── AES key (cached globally — one key shared by all cameras) ─────────────

    def get_key(self, key_url: str) -> bytes:
        with self._key_lock:
            if self._aes_key is None:
                abs_url = key_url if key_url.startswith("http") else (
                    settings.sentinel_base_url + key_url
                )
                resp = self._session.get(abs_url, timeout=10)
                resp.raise_for_status()
                self._aes_key = resp.content
                log.debug("AES key fetched (%d bytes)", len(self._aes_key))
            return self._aes_key

    # ── segment download + decrypt ────────────────────────────────────────────

    def fetch_segment(self, seg_url: str, key_url: str) -> bytes:
        """Download a .ts segment, decrypt it, and return raw MPEG-TS bytes."""
        resp = self._session.get(seg_url, timeout=90)
        resp.raise_for_status()
        encrypted = resp.content

        key = self.get_key(key_url)
        iv = bytes(16)  # IV is all-zeros in this gateway's playlist
        return AES.new(key, AES.MODE_CBC, iv).decrypt(encrypted)


# ─── CameraStream ─────────────────────────────────────────────────────────────

class CameraStream:
    """
    Unified stream abstraction that yields decoded frames.

    Usage::

        with CameraStream(cam_id, cookie) as stream:
            for frame_idx, pts_ms, frame_bgr in stream.frames():
                ...  # frame_bgr is a numpy array (H, W, 3) BGR

    Connection strategy:
    - RTSP (with email+password in URL) if port 8554 is reachable
    - HLS manual-decrypt loop otherwise
    """

    def __init__(self, cam_id: str, cookie: str) -> None:
        self.cam_id = cam_id
        self.cookie = cookie
        self.cap: Optional[cv2.VideoCapture] = None
        self.use_rtsp = is_rtsp_reachable(cam_id)
        self.prefix = f"[{cam_id}]"

        mode = "RTSP" if self.use_rtsp else "HLS"
        url_display = settings.rtsp_url(cam_id) if self.use_rtsp else settings.hls_url(cam_id)
        log.info("%s Using %s — %s", self.prefix, mode, url_display)

    # ── RTSP path ─────────────────────────────────────────────────────────────

    def _rtsp_connect(self) -> None:
        if self.cap:
            self.cap.release()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(settings.rtsp_url(self.cam_id), cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream for {self.cam_id}")

    def _frames_rtsp(self) -> Generator:
        backoff = settings.stream_reconnect_base_s
        frame_idx = 0
        while True:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._rtsp_connect()
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    log.warning("%s RTSP read failed, reconnecting in %.0fs...", self.prefix, backoff)
                    time.sleep(backoff)
                    backoff = min(settings.stream_reconnect_max_s, backoff * 2)
                    self._rtsp_connect()
                    continue
                backoff = settings.stream_reconnect_base_s  # reset on success
                if frame_idx % settings.frame_skip == 0:
                    yield frame_idx, self.cap.get(cv2.CAP_PROP_POS_MSEC), frame
                frame_idx += 1
            except Exception as exc:
                log.error("%s RTSP error: %s", self.prefix, exc)
                time.sleep(backoff)
                backoff = min(settings.stream_reconnect_max_s, backoff * 2)
                self.cap = None

    # ── HLS path ──────────────────────────────────────────────────────────────

    def _frames_hls(self) -> Generator:
        fetcher = _HLSFetcher(self.cookie)
        frame_idx = 0
        backoff = settings.stream_reconnect_base_s

        while True:
            try:
                log.info("%s Fetching HLS manifest...", self.prefix)
                segments, key_url = fetcher.fetch_manifest(self.cam_id)
                if not segments:
                    log.warning("%s Empty segment list, retrying...", self.prefix)
                    time.sleep(backoff)
                    continue

                log.info("%s %d segments in archive", self.prefix, len(segments))
                backoff = settings.stream_reconnect_base_s  # reset

                # ── Double-buffered segment download ──────────────────────────
                # While we decode segment N, we prefetch segment N+1 in the
                # background so there is no pause between segments for the viewer.

                prefetch_data: list[Optional[bytes]] = [None]
                prefetch_event = threading.Event()

                def _prefetch(url: str) -> None:
                    try:
                        prefetch_data[0] = fetcher.fetch_segment(url, key_url)
                    except Exception as ex:
                        log.warning("%s Prefetch failed for %s: %s", self.prefix, url, ex)
                        prefetch_data[0] = None
                    finally:
                        prefetch_event.set()

                # Kick off first segment download immediately
                threading.Thread(target=_prefetch, args=(segments[0],), daemon=True).start()

                for seg_idx, seg_url in enumerate(segments):
                    # Wait for this segment's prefetch to complete
                    if not prefetch_event.wait(timeout=120):
                        log.warning("%s Segment %d timed out, skipping", self.prefix, seg_idx)
                    seg_bytes = prefetch_data[0]
                    prefetch_event.clear()
                    prefetch_data[0] = None

                    # Start prefetching the next segment right away
                    if seg_idx + 1 < len(segments):
                        threading.Thread(
                            target=_prefetch, args=(segments[seg_idx + 1],), daemon=True
                        ).start()

                    if seg_bytes is None:
                        continue  # download failed — skip to next

                    # Write decrypted .ts to a temp file and decode with OpenCV
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ts")
                    try:
                        with os.fdopen(tmp_fd, "wb") as f:
                            f.write(seg_bytes)

                        cap = cv2.VideoCapture(tmp_path)
                        seg_frames = 0
                        while True:
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                break
                            if seg_frames % settings.frame_skip == 0:
                                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                                yield frame_idx, pts_ms, frame
                            seg_frames += 1
                            frame_idx += 1
                        cap.release()

                        log.debug(
                            "%s Segment %d/%d — %d frames decoded",
                            self.prefix, seg_idx + 1, len(segments), seg_frames,
                        )
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                log.info("%s HLS archive complete — looping from start", self.prefix)

            except Exception as exc:
                log.error("%s HLS error: %s", self.prefix, exc)
                time.sleep(backoff)
                backoff = min(settings.stream_reconnect_max_s, backoff * 2)

    # ── Public interface ──────────────────────────────────────────────────────

    def frames(self) -> Generator:
        """Yield (frame_idx, pts_ms, frame_bgr) indefinitely."""
        if self.use_rtsp:
            yield from self._frames_rtsp()
        else:
            yield from self._frames_hls()

    def __enter__(self) -> "CameraStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cap:
            self.cap.release()
