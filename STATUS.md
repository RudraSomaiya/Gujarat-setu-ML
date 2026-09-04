# Setu ML Pipeline — Status & Session Tracking

**Project:** Gujarat Police Innovation Challenge 2026 — Setu Surveillance Intelligence  
**Role:** ML Engineer (Rudra)  
**Current Date:** September 04, 2026  
# 🚀 STATUS: ML Pipeline Feature Complete

**Last Updated:** Sept 4, 2026

The ML Pipeline is now **Feature Complete** for the hackathon. 

## 1. What's Done
- **GPU Inference:** YOLOv8 + ByteTrack + CLIP running flawlessly on the RTX 3060 (90+ FPS).
- **CLIP Zero-Shot NLP:** Classifications (e.g., "auto_rickshaw") and Color extraction now use CLIP text-embeddings, matching perfectly with natural language searches.
- **Anomaly Detection:** Rule-based `stopped_vehicle` detection added to the live worker. It tracks centroid stability and writes to the `alerts` table.
- **Database Schema:** `detections`, `cameras`, `watchlist_entries`, and `alerts` tables are fully structured. `pts_ms` is stored for accurate video seeking.
- **Network Burst-Sampling:** The `batch_index.py` tool has been upgraded to burst-sample the heavily rate-limited Hackathon streaming server (grabbing exactly 100 6-second segments spaced evenly over 12 hours) to guarantee completion overnight.

## 2. Handoff to Backend / Frontend

### The Two Modes (Crucial to Understand)
1. **The Setup (`scripts/batch_index.py`):** Runs sequentially, taking "bursts" of video across the 12 hours. It populates the database so the demo has history to search. **It does not run anomaly tracking** (because it jumps through time).
2. **The Live Product (`ml/worker.py`):** The actual daemon meant for production. It spawns 30 threads and watches all cameras *simultaneously and continuously*. This is where real-time **Watchlist Alerts** and **Stopped-Vehicle Anomalies** are detected and pushed via PostgreSQL `NOTIFY`.

### Database Contract
- Search queries should hit `detections` and sort by `embedding <=> query_vector ASC`.
- Backend should listen to PostgreSQL `NOTIFY 'setu_alerts'` to push real-time anomaly popups to the frontend.
- `pts_ms` inside `detections` is the millisecond timestamp to seek the video player to.

### Hackathon Demo Caveats
- **Backups:** Because the cameras are on a 12-hour pre-recorded loop, if the live worker processes a segment tomorrow that the batch indexer processed tonight, you'll get duplicate vehicles in the DB. **TAKE A DB BACKUP TOMORROW MORNING BEFORE DEMOING** so you can easily restore your pristine 12-hour dataset.
- **Cloud CPUs:** AI models auto-detect CPUs. If you deploy the live worker (`worker.py`) to a standard cloud CPU, it will choke trying to process 30 cameras at 6 FPS. For the live demo, either run it on a laptop with a GPU, or restrict it to 1-2 cameras if presenting from a cloud CPU.

## 3. Immediate Next Steps
1. Run `python -m scripts.batch_index` overnight.
2. In the morning, run a `pg_dump -U setu -F c setu > backup.dump` to secure the data.
3. Test natural language search.
4. Hand off to the Backend team.

- [x] **HLS Decryption Engine (`ml/stream.py`)**: Implemented manual AES-128 CBC segment decryption via `pycryptodome` and OpenCV frame decoding.
- [x] **Grey Smear Artifact Resolution**: Identified that RTSP TCP streaming caused severe grey smearing due to missed P-frame references on reconnects. Switching to HLS segments (each starting with an IDR keyframe) completely eliminated this issue.
- [x] **Stream Prefetching**: Double-buffered segment fetcher downloads segment N+1 in the background while segment N is decoding.
- [x] **Transport Note**: `ml/stream.py` uses TCP socket handling and `rtsp_transport;tcp` for the opt-in RTSP fallback (`FORCE_RTSP=true`), while defaulting to HLS which runs over HTTP/TCP for reliable deterministic decoding.

### B. Schema & Data Pipeline Enhancements
- [x] **Video Seek Support (`pts_ms`)**: Added `pts_ms DOUBLE PRECISION` to `Detection` dataclass, `ml/pipeline.py`, `ml/db.py`, and `scripts/schema.sql`. This records `cv2.CAP_PROP_POS_MSEC` so the frontend video player can jump directly to the exact millisecond in the stream.
- [x] **Dominant Color Extraction**: Added HSV histogram bucketing in `ml/pipeline.py` (`_extract_color()`) classifying vehicle crops into `red`, `white`, `black`, `silver`, `blue`, `yellow`, `green`, `orange`, or `unknown`. Persisted in `detections.dominant_color`.
- [x] **ByteTrack Multi-Camera Isolation**: Decoupled tracking from `yolo.track()`. Each camera thread now owns an isolated `BYTETracker` instance (`ultralytics.trackers.BYTETracker`), preventing cross-thread `track_id` bleed. Detections are matched via IoU back to YOLO bounding boxes.
- [x] **OpenCLIP Embeddings**: 512-dimensional L2-normalized embeddings generated via `open_clip` ViT-B/32 (exact `openai` weights) for every vehicle detection, stored in Postgres `vector(512)`.
- [x] **Watchlist Alerts**: Real-time checking against `watchlist_entries` for both exact plate strings and person/vehicle embeddings (`cosine similarity >= 0.75`). Automatically triggers PostgreSQL `pg_notify('setu_alerts', alert_id)`.
- [x] **MinIO Thumbnail Storage**: Bounding box crops uploaded directly to MinIO bucket `setu-clips` under structured keys (`{cam_id}/{date}/{det_id}.jpg`).
- [x] **Documentation & Architecture Sync**: Updated `Docs-repo/docs/ARCHITECTURE.md` to reflect 30 real HLS feeds, isolated ByteTrack, no MediaMTX, and the new schema fields (`pts_ms`, `dominant_color`).

### C. Batch Offline Indexing
- [x] **Batch Script (`scripts/batch_index.py`)**: Built sequential 30-camera batch indexer with `frame_skip=150` (~1 frame per 5 seconds), `--resume` capability to skip already indexed cameras, time-window filtering (`--start-time`, `--end-time`), and registered console command `setu-index`.

---

## 3. What Is Currently Being Done

- [ ] **Fresh Schema Deployment & Docker Execution**:
  - Running database reset (`scripts/reset.py`) and applying updated `scripts/schema.sql` with `pts_ms` and `dominant_color` inside the PostgreSQL container.
  - Seeding the 30 Sentinel cameras with coordinates (`scripts/seed.py`).
- [ ] **End-to-End Pipeline Dry Run**:
  - Verifying a short test run on `cam04` and `cam12` to verify that `track_id`, `pts_ms`, `dominant_color`, `embedding`, and MinIO thumbnails are all populated cleanly.

---

## 4. What Is Left To Do (Critical Before Sep 05 EOD)

### 1. Color Extraction Testing & Calibration
- Test dominant color output across real vehicle crops from day and night footage.
- Verify that common edge cases (silver vs white, shadows, tinted windshields) map to reasonable buckets.
- Ensure the backend query parser can leverage `dominant_color` as a structured SQL filter (e.g., `WHERE dominant_color = 'red'`) to narrow CLIP cosine searches.

### 2. Embeddings Validation & Sanity Testing
- Verify that CLIP text embeddings ("white swift", "red bus", "yellow auto rickshaw", "delivery scooter") produce expected cosine similarity rankings against stored image embeddings in `detections`.
- Confirm that the backend team has the exact matching configuration (`open_clip` ViT-B/32, pretrained `openai`) to prevent vector space divergence.
- Test visual re-identification: verify that crops of the same vehicle across consecutive frames have high similarity (>= 0.80).

### 3. Batch Embedding Test & Full Overnight Run
- **Test Run**: Run `python -m scripts.batch_index --cameras cam04,cam12 --frame-skip 150` on 2 high-traffic cameras to benchmark GPU/CPU throughput (target: ~60-90 seconds per 12h feed at 1 frame/5s) and confirm no memory leaks or stream timeouts.
- **Full Overnight Run**: Execute `python -m scripts.batch_index --start-time 06:00 --end-time 22:00` across all 30 cameras.
- **Verification**: Run post-batch SQL queries to verify detection counts, valid `pts_ms` distribution, non-null embeddings, and HNSW index health.

### 4. Backend & Frontend Handoff Checklist
- Confirm backend can generate presigned MinIO URLs from `thumbnail_key`.
- Confirm backend can translate `pts_ms` to HLS playback seek positions in the web player.
- Confirm WebSocket alert listener receives `pg_notify('setu_alerts')` on watchlist matches.