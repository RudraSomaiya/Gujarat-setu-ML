import uuid
import logging
from dataclasses import dataclass
from typing import Optional
import cv2
from datetime import datetime, timezone
from PIL import Image
import torch
import numpy as np
import re

from ml.config import settings
from ml.models import get_yolo, get_clip, get_plate_detector, get_plate_ocr, VEHICLE_CLASSES, device

log = logging.getLogger(__name__)

import threading

_inference_lock = threading.RLock()

@dataclass
class Detection:
    id: str
    camera_id: str
    track_id: Optional[int]
    class_name: str
    bbox: dict
    confidence: float
    plate_text: Optional[str]
    plate_confidence: Optional[float]
    embedding: list[float]
    thumbnail_key: Optional[str]
    ts: datetime

class DetectionPipeline:
    def __init__(self):
        self.yolo = get_yolo()
        self.clip, self.clip_preprocess = get_clip()
        self.plate_detector = get_plate_detector()
        self.plate_ocr = get_plate_ocr()

    def process_frame(self, frame_bgr: np.ndarray, camera_id: str, pts_ms: float) -> list[Detection]:
        with _inference_lock:
            results = self.yolo.predict(
                frame_bgr, 
                classes=list(VEHICLE_CLASSES.keys()), 
                conf=settings.detection_conf_threshold, 
                verbose=False, 
            )
        
        detections = []
        if not results or not results[0].boxes:
            return detections
            
        boxes = results[0].boxes
        ts = datetime.now(timezone.utc)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            # bounds check
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
                
            embedding = self._embed_crop(crop)
            if not embedding:
                continue
                
            plate_text, plate_conf = None, None
            if cls_id in [2, 3, 5, 7]: # vehicles
                plate_text, plate_conf = self._extract_plate(crop)
                
            detections.append(Detection(
                id=str(uuid.uuid4()),
                camera_id=camera_id,
                track_id=track_id,
                class_name=VEHICLE_CLASSES.get(cls_id, 'unknown'),
                bbox={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                confidence=conf,
                plate_text=plate_text,
                plate_confidence=plate_conf,
                embedding=embedding,
                thumbnail_key=None,
                ts=ts
            ))
            
        return detections

    def _embed_crop(self, crop_bgr: np.ndarray) -> list[float]:
        try:
            crop_rgb = crop_bgr[:, :, ::-1]
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                embed = self.clip.encode_image(img_tensor)
                embed /= embed.norm(dim=-1, keepdim=True)
            return embed[0].cpu().tolist()
        except Exception:
            return []

    def _extract_plate(self, vehicle_crop: np.ndarray) -> tuple[Optional[str], Optional[float]]:
        try:
            res = self.plate_detector(vehicle_crop, verbose=False, stream=False)
            if not res or not res[0].boxes:
                return None, None
                
            boxes = res[0].boxes
            best_idx = int(boxes.conf.argmax())
            plate_conf = float(boxes.conf[best_idx])
            
            if plate_conf < settings.plate_conf_threshold:
                return None, None
                
            px1, py1, px2, py2 = map(int, boxes.xyxy[best_idx].tolist())
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            
            if plate_crop.size == 0:
                return None, None
                
            plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            preds = self.plate_ocr.run(plate_crop_gray, return_confidence=True)
            if not preds:
                return None, None
                
            pred = preds[0]
            clean_text = re.sub(r'[^A-Z0-9]', '', pred.plate.upper())
            if len(clean_text) < 4:
                return None, None
                
            ocr_conf = float(sum(pred.char_probs) / len(pred.char_probs)) if pred.char_probs is not None and len(pred.char_probs) > 0 else 0.0
            return clean_text, min(plate_conf, ocr_conf)
        except Exception as e:
            log.warning(f"Plate extraction failed: {e}")
            return None, None
