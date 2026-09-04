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
    pts_ms: float           # position-in-segment in milliseconds (from CAP_PROP_POS_MSEC)
    dominant_color: Optional[str]  # HSV colour bucket for NL search precision

class DetectionPipeline:
    # ── CLIP zero-shot label sets ─────────────────────────────────────────────
    # These text prompts are encoded once at init time. For each detection crop
    # the image embedding is compared (dot product) against each label embedding
    # and the argmax becomes the label. Cost: 8 dot products per detection.
    # Both color and class use this approach — no HSV math, no COCO-class limits.

    _CLASS_PROMPTS = [
        "auto rickshaw",       # India-specific — COCO doesn't have this
        "motorcycle or bike",
        "city bus or coach",
        "truck or lorry",
        "car or sedan or SUV",
        "tractor or farm vehicle",
        "pickup truck or van",
        "person walking",
    ]
    _CLASS_NAMES = [
        "auto_rickshaw", "motorcycle", "bus", "truck",
        "car", "tractor", "truck", "person",
    ]

    _COLOR_PROMPTS = [
        "a red coloured vehicle",
        "a blue coloured vehicle",
        "a white coloured vehicle",
        "a yellow coloured vehicle",
        "a black coloured vehicle",
        "a silver or grey coloured vehicle",
        "a green coloured vehicle",
        "an orange coloured vehicle",
    ]
    _COLOR_NAMES = [
        "red", "blue", "white", "yellow", "black", "silver", "green", "orange",
    ]

    def __init__(self):
        self.yolo = get_yolo()
        self.clip, self.clip_preprocess = get_clip()
        self.plate_detector = get_plate_detector()
        self.plate_ocr = get_plate_ocr()

        # Pre-encode CLIP label sets once at startup.
        # Shape: (N_labels, 512), L2-normalized — ready for dot-product scoring.
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            cls_tokens = tokenizer(self._CLASS_PROMPTS).to(device)
            cls_embs = self.clip.encode_text(cls_tokens)
            cls_embs /= cls_embs.norm(dim=-1, keepdim=True)
            self._class_label_embs = cls_embs  # (8, 512) on device

            col_tokens = tokenizer(self._COLOR_PROMPTS).to(device)
            col_embs = self.clip.encode_text(col_tokens)
            col_embs /= col_embs.norm(dim=-1, keepdim=True)
            self._color_label_embs = col_embs  # (8, 512) on device

        # One isolated ByteTracker per DetectionPipeline instance.
        # worker.py creates one pipeline per camera thread, so this is effectively
        # one tracker per camera — no shared state, no cross-camera track_id bleed.
        from ultralytics.trackers import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace
        _tracker_args = IterableSimpleNamespace(
            track_high_thresh=0.5,    # min conf to enter high-conf association
            track_low_thresh=0.1,     # min conf for low-conf second-pass association
            new_track_thresh=0.6,     # min conf to create a new track
            track_buffer=30,          # frames to keep a lost track (30 ≈ 5s at 6 FPS)
            match_thresh=0.8,         # IoU threshold for first-pass association
            fuse_score=True,          # fuse detection score into track score
            frame_rate=6,             # approximate processed FPS (frame_skip=5 at 30 FPS)
        )
        self.tracker = BYTETracker(_tracker_args)

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

        yolo_result = results[0]
        boxes = yolo_result.boxes
        ts = datetime.now(timezone.utc)

        # ── ByteTrack: feed YOLO boxes into this camera's isolated tracker ────
        # tracker.update() returns an (N, 7+) array: [x1,y1,x2,y2,track_id,conf,cls_id]
        # Each row's original index into `boxes` is stored in row[-1] (the `idx` field).
        # We build a dict {box_idx: track_id} for fast lookup below.
        track_id_by_idx: dict[int, int] = {}
        try:
            tracked = self.tracker.update(boxes.cpu(), frame_bgr)
            if tracked is not None and len(tracked) > 0:
                # tracked array columns: x1,y1,x2,y2, track_id, conf, cls, *extra
                # The tracker preserves the detection index in an attribute on STrack;
                # the returned numpy array has it at column index 4 as the track ID.
                # Ultralytics BYTETracker returns shape (N,7): xyxy + tid + conf + cls
                for row in tracked:
                    if len(row) >= 7:
                        # Find matching box by IoU with x1y1x2y2
                        tx1, ty1, tx2, ty2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        tid = int(row[4])
                        # Match back to original box index by closest xyxy
                        best_idx, best_iou = -1, -1.0
                        for i, box in enumerate(boxes):
                            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                            inter_x1, inter_y1 = max(tx1, bx1), max(ty1, by1)
                            inter_x2, inter_y2 = min(tx2, bx2), min(ty2, by2)
                            inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                            union = ((tx2-tx1)*(ty2-ty1) + (bx2-bx1)*(by2-by1) - inter)
                            iou = inter / union if union > 0 else 0.0
                            if iou > best_iou:
                                best_iou, best_idx = iou, i
                        if best_idx >= 0 and best_iou > 0.3:
                            track_id_by_idx[best_idx] = tid
        except Exception as e:
            log.warning("ByteTrack update failed: %s — falling back to no tracking", e)
        # ─────────────────────────────────────────────────────────────────────

        h, w = frame_bgr.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = track_id_by_idx.get(i)  # None if tracker didn't assign this box
            
            # bounds check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            embedding_tensor, embedding = self._embed_crop(crop)
            if not embedding:
                continue

            # ── CLIP zero-shot color + class ──────────────────────────────────
            # embedding_tensor is shape (1, 512), already L2-normalised, on device.
            # Label embs are (N, 512). Dot product = cosine similarity.
            with torch.no_grad():
                color_scores = (embedding_tensor @ self._color_label_embs.T)[0]
                color_idx = int(color_scores.argmax().item())
                dominant_color = self._COLOR_NAMES[color_idx]

                class_scores = (embedding_tensor @ self._class_label_embs.T)[0]
                class_idx = int(class_scores.argmax().item())
                clip_class_name = self._CLASS_NAMES[class_idx]
            # ──────────────────────────────────────────────────────────────────

            plate_text, plate_conf = None, None
            # Run plate OCR on all detections; CLIP class may catch vehicles YOLO
            # labelled as 'person' or vice-versa, so don't gate on cls_id alone.
            plate_text, plate_conf = self._extract_plate(crop)

            detections.append(Detection(
                id=str(uuid.uuid4()),
                camera_id=camera_id,
                track_id=track_id,
                class_name=clip_class_name,   # CLIP zero-shot, not YOLO COCO label
                bbox={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                confidence=conf,
                plate_text=plate_text,
                plate_confidence=plate_conf,
                embedding=embedding,
                thumbnail_key=None,
                ts=ts,
                pts_ms=pts_ms,
                dominant_color=dominant_color,  # CLIP zero-shot, not HSV buckets
            ))

        return detections

    def _embed_crop(self, crop_bgr: np.ndarray) -> tuple[torch.Tensor | None, list[float]]:
        """Encode a BGR crop with CLIP. Returns (tensor, list) — tensor stays on
        device for zero-shot label scoring; list is stored in DB."""
        try:
            crop_rgb = crop_bgr[:, :, ::-1]
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                embed = self.clip.encode_image(img_tensor)
                embed /= embed.norm(dim=-1, keepdim=True)
            return embed, embed[0].cpu().tolist()
        except Exception:
            return None, []

    def _extract_color(self, crop_bgr: np.ndarray) -> Optional[str]:
        """Dominant colour of the crop using HSV histogram bucketing.

        Returns a colour name string suitable for NL search filtering:
        red | white | black | silver | blue | yellow | green | orange | unknown
        """
        try:
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0].astype(np.float32)  # 0–179 in OpenCV
            s = hsv[:, :, 1].astype(np.float32)  # 0–255
            v = hsv[:, :, 2].astype(np.float32)  # 0–255

            total = h.size
            if total == 0:
                return "unknown"

            # ── Achromatic buckets (low saturation) ───────────────────────
            low_s_mask = s < 50   # unsaturated pixels
            low_s_frac = low_s_mask.sum() / total

            if low_s_frac > 0.55:
                # Mostly achromatic — classify by brightness
                mean_v = float(v[low_s_mask].mean()) if low_s_mask.any() else float(v.mean())
                if mean_v < 60:
                    return "black"
                elif mean_v < 140:
                    return "silver"
                else:
                    return "white"

            # ── Chromatic: use hue histogram on saturated pixels only ─────
            sat_mask = s >= 50
            if not sat_mask.any():
                return "unknown"

            h_sat = h[sat_mask]

            # OpenCV hue: 0–179 (degrees / 2). Bucket boundaries in that space:
            #   red:    [0,10) ∪ [170,180)
            #   orange: [10,20)
            #   yellow: [20,35)
            #   green:  [35,85)
            #   blue:   [85,130)
            #   (purple/violet skipped — rare in vehicles)
            buckets = {
                "red":    ((h_sat < 10) | (h_sat >= 170)).sum(),
                "orange": ((h_sat >= 10) & (h_sat < 20)).sum(),
                "yellow": ((h_sat >= 20) & (h_sat < 35)).sum(),
                "green":  ((h_sat >= 35) & (h_sat < 85)).sum(),
                "blue":   ((h_sat >= 85) & (h_sat < 130)).sum(),
            }
            dominant = max(buckets, key=lambda k: buckets[k])
            return dominant
        except Exception:
            return "unknown"

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
            
            # ── Preprocessing: upscale + CLAHE + sharpen ──
            h, w = plate_crop.shape[:2]
            upscaled = cv2.resize(plate_crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
            plate_processed = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
            
            preds = self.plate_ocr.run(plate_processed, return_confidence=True)
            if not preds:
                return None, None
                
            pred = preds[0]
            clean_text = re.sub(r'[^A-Z0-9]', '', pred.plate.upper())
            if len(clean_text) < 4:
                return None, None
                
            ocr_conf = float(sum(pred.char_probs) / len(pred.char_probs)) if pred.char_probs is not None and len(pred.char_probs) > 0 else 0.0
            
            # ── Indian plate regex validation + fuzzy correction ──
            from ml.plate_validator import validate_plate
            validated = validate_plate(clean_text)
            if validated:
                return validated, min(plate_conf, ocr_conf)
            
            # Return raw text if validation fails (still useful for debugging)
            return clean_text, min(plate_conf, ocr_conf)
        except Exception as e:
            log.warning(f"Plate extraction failed: {e}")
            return None, None
