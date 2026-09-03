"""
OCR model comparison script.

Captures frames from HLS, detects license plates, then runs 3 OCR engines
on each detected plate crop and compares results:
  1. fast-plate-ocr (current)
  2. EasyOCR
  3. fast-plate-ocr + preprocessing

Also applies Indian plate regex validation + fuzzy correction to each.

Usage:
    python -m scripts.compare_ocr [cam_id] [--frames N]
"""

import argparse
import os
import sys
import cv2
import numpy as np
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from ml.models import get_yolo, get_plate_detector, get_plate_ocr, VEHICLE_CLASSES, warm_all_models
from ml.auth import get_cookie
from ml.stream import CameraStream
from ml.config import settings
from ml.plate_validator import validate_plate, correct_plate


def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing for plate crops before OCR.
    
    Steps:
    1. Upscale 3x with cubic interpolation (more detail for OCR)
    2. Convert to grayscale
    3. CLAHE adaptive histogram equalization (handles uneven lighting)
    4. Bilateral filter (smooth noise while preserving edges)
    5. Sharpen with unsharp mask
    """
    # 1. Upscale 3x
    h, w = plate_bgr.shape[:2]
    upscaled = cv2.resize(plate_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (adaptive contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. Bilateral filter — reduce noise while keeping edges sharp
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 5. Unsharp mask (sharpen)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
    
    return sharpened


def run_fast_plate_ocr(plate_crop_bgr: np.ndarray, ocr_model, preprocess: bool = False):
    """Run fast-plate-ocr on a plate crop."""
    if preprocess:
        gray = preprocess_plate(plate_crop_bgr)
    else:
        gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
    
    preds = ocr_model.run(gray, return_confidence=True)
    if not preds:
        return None, 0.0
    pred = preds[0]
    text = re.sub(r'[^A-Z0-9]', '', pred.plate.upper())
    conf = float(sum(pred.char_probs) / len(pred.char_probs)) if pred.char_probs is not None and len(pred.char_probs) > 0 else 0.0
    return text, conf


def run_easyocr(plate_crop_bgr: np.ndarray, reader, preprocess: bool = True):
    """Run EasyOCR on a plate crop."""
    if preprocess:
        gray = preprocess_plate(plate_crop_bgr)
    else:
        gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # EasyOCR expects BGR or grayscale
    results = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if not results:
        return None, 0.0
    
    # Concatenate all detected text regions
    texts = []
    confs = []
    for (bbox, text, conf) in results:
        texts.append(text.upper())
        confs.append(conf)
    
    full_text = re.sub(r'[^A-Z0-9]', '', ''.join(texts))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_text, avg_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_id", nargs="?", default="cam06")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--save-crops", action="store_true", help="Save plate crops to disk")
    args = parser.parse_args()
    
    # Create output dir for crops
    crop_dir = Path("plate_crops")
    if args.save_crops:
        crop_dir.mkdir(exist_ok=True)
    
    log.info("Loading models...")
    yolo = get_yolo()
    plate_det = get_plate_detector()
    fast_ocr = get_plate_ocr()
    
    # Load EasyOCR
    log.info("Loading EasyOCR...")
    try:
        import easyocr
        easy_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        has_easyocr = True
        log.info("EasyOCR loaded ✓")
    except ImportError:
        log.warning("EasyOCR not installed — skipping")
        has_easyocr = False
        easy_reader = None
    
    cookie = get_cookie()
    
    log.info(f"\n{'='*80}")
    log.info(f"Comparing OCR models on {args.camera_id} — processing {args.frames} frames")
    log.info(f"{'='*80}\n")
    
    plate_count = 0
    frame_count = 0
    
    with CameraStream(args.camera_id, cookie) as stream:
        for frame_idx, pts_ms, frame in stream.frames():
            if frame_count >= args.frames:
                break
            frame_count += 1
            
            # Detect vehicles
            results = yolo.predict(
                frame, classes=list(VEHICLE_CLASSES.keys()),
                conf=settings.detection_conf_threshold, verbose=False
            )
            if not results or not results[0].boxes:
                continue
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in [2, 3, 5, 7]:  # vehicles only
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size == 0:
                    continue
                
                # Detect plate within vehicle
                plate_res = plate_det(vehicle_crop, verbose=False, stream=False)
                if not plate_res or not plate_res[0].boxes:
                    continue
                
                pboxes = plate_res[0].boxes
                best_idx = int(pboxes.conf.argmax())
                plate_conf = float(pboxes.conf[best_idx])
                
                if plate_conf < 0.20:  # low threshold to get more samples
                    continue
                
                px1, py1, px2, py2 = map(int, pboxes.xyxy[best_idx].tolist())
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 5 or plate_crop.shape[1] < 10:
                    continue
                
                plate_count += 1
                
                # Save crop if requested
                if args.save_crops:
                    cv2.imwrite(str(crop_dir / f"plate_{plate_count:04d}.png"), plate_crop)
                    cv2.imwrite(str(crop_dir / f"plate_{plate_count:04d}_preprocessed.png"), 
                               preprocess_plate(plate_crop))
                
                # ── Run all OCR models ──
                
                # 1. fast-plate-ocr (raw, no preprocessing)
                fp_raw, fp_raw_conf = run_fast_plate_ocr(plate_crop, fast_ocr, preprocess=False)
                
                # 2. fast-plate-ocr (with preprocessing)
                fp_pre, fp_pre_conf = run_fast_plate_ocr(plate_crop, fast_ocr, preprocess=True)
                
                # 3. EasyOCR (with preprocessing)
                if has_easyocr:
                    easy_text, easy_conf = run_easyocr(plate_crop, easy_reader, preprocess=True)
                else:
                    easy_text, easy_conf = "N/A", 0.0
                
                # ── Validate each ──
                fp_raw_valid = validate_plate(fp_raw) if fp_raw else None
                fp_pre_valid = validate_plate(fp_pre) if fp_pre else None
                easy_valid = validate_plate(easy_text) if easy_text and easy_text != "N/A" else None
                
                cls_name = VEHICLE_CLASSES.get(cls_id, "?")
                
                log.info(f"── Plate #{plate_count} ({cls_name}, det={plate_conf:.0%}) ──")
                log.info(f"  fast-plate-ocr (raw):    {fp_raw or '-':>14s}  ({fp_raw_conf:.0%})  → {fp_raw_valid or '✗ invalid'}")
                log.info(f"  fast-plate-ocr (pre):    {fp_pre or '-':>14s}  ({fp_pre_conf:.0%})  → {fp_pre_valid or '✗ invalid'}")
                if has_easyocr:
                    log.info(f"  EasyOCR      (pre):      {easy_text or '-':>14s}  ({easy_conf:.0%})  → {easy_valid or '✗ invalid'}")
                log.info("")
    
    log.info(f"{'='*80}")
    log.info(f"Done. Processed {frame_count} frames, found {plate_count} plate crops.")
    if args.save_crops:
        log.info(f"Crops saved to {crop_dir.absolute()}")


if __name__ == "__main__":
    main()
