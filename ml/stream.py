import logging
import time
import socket
from contextlib import contextmanager
import cv2
import os

from ml.config import settings

log = logging.getLogger(__name__)

def is_rtsp_reachable(cam_id: str) -> bool:
    try:
        s = socket.create_connection(("103.250.160.189", 8554), timeout=2)
        s.close()
        return True
    except OSError:
        return False

import threading

class CameraStream:
    def __init__(self, cam_id: str, cookie: str):
        self.cam_id = cam_id
        self.cookie = cookie
        self.cap = None
        self.use_rtsp = is_rtsp_reachable(cam_id)
        self.url = settings.rtsp_url(cam_id) if self.use_rtsp else settings.hls_url(cam_id)
        self.prefix = f"[{self.cam_id}]"

    def _connect(self):
        if self.cap:
            self.cap.release()
        
        if self.use_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        else:
            from ml.auth import get_ffmpeg_options
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = get_ffmpeg_options(self.cookie)
            
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open stream for {self.cam_id}")

    def frames(self):
        backoff = settings.stream_reconnect_base_s
        frame_idx = 0
        
        while True:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._connect()
                    
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    log.warning(f"{self.prefix} Stream read failed, reconnecting in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(settings.stream_reconnect_max_s, backoff * 2)
                    self._connect()
                    continue
                    
                backoff = settings.stream_reconnect_base_s # reset
                
                # frame skip
                if frame_idx % settings.frame_skip == 0:
                    pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    yield frame_idx, pts_ms, frame
                    
                frame_idx += 1
                
            except Exception as e:
                log.error(f"{self.prefix} Stream error: {e}")
                time.sleep(backoff)
                backoff = min(settings.stream_reconnect_max_s, backoff * 2)
                self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
