# Setu — ML Worker

AI inference pipeline for cross-camera vehicle tracking, ANPR, and natural-language search.
Part of the Setu (Model 2/3) AI surveillance layer for the Gujarat Police Innovation Challenge 2026.

## What it does

Connects to the Sentinel CCTV sandbox (30 real Gujarat camera feeds), runs AI inference on each frame, and writes structured data to PostgreSQL + MinIO:

| Step | Model | Output |
|---|---|---|
| Detection | YOLOv8s (COCO) | bounding boxes per frame |
| Tracking | ByteTrack (built into Ultralytics) | consistent `track_id` across frames |
| Embedding | open_clip ViT-B/32 (OpenAI weights) | 512-dim vector per detection |
| ANPR | yolov8n-license-plate + fast-plate-ocr | `plate_text` or `null` |
| Persistence | psycopg2 + pgvector | `detections` rows in Postgres |
| Thumbnails | minio | JPEG crops in `setu-clips` bucket |
| Alerts | SQL | `alerts` row + `pg_notify` when watchlist matches |

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (RTX 3060 or better) — CPU fallback works but is slow
- Docker (for Postgres + MinIO)
- Access to the Sentinel camera sandbox (`DBAD-4UQP-BSTV`)

## Setup

```bash
cd ml/

# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install PyTorch with CUDA first (large download, do this once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -e .

# 4. Copy env file
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# 5. Start local Postgres (pgvector) + MinIO
docker run -d --name setu-pg -p 5432:5432 \
  -e POSTGRES_USER=setu -e POSTGRES_PASSWORD=setu_dev_only -e POSTGRES_DB=setu \
  pgvector/pgvector:pg16

docker run -d --name setu-minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=setu -e MINIO_ROOT_PASSWORD=setu_dev_only \
  minio/minio server /data --console-address ":9001"

# 6. Apply DB schema
psql postgresql://setu:setu_dev_only@localhost:5432/setu -f scripts/schema.sql

# 7. Seed cameras + watchlist
python -m ml.scripts.seed

# 8. (Optional) Check which feeds are reachable
python -m ml.scripts.check_feeds
```

## Run

```bash
# Start the multi-camera worker
python -m ml.worker

# Or the installed entry point:
setu-worker
```

Set `CAMERAS=cam04,cam05,cam06` in `.env` to choose which cameras to process.
Set `CAMERAS=all` to process all 30 (GPU required).

## Docker

```bash
# Build (pre-downloads all AI model weights into the image)
docker build -t setu-ml .

# Run with GPU
docker run --gpus all --env-file .env --network host setu-ml
```

## Architecture notes

- **RTSP primary, HLS fallback**: RTSP (`rtsp://103.250.160.189:8554/stream/{id}`) needs no auth and is lower-latency. HLS (`https://cctv.corp8.cloud/{id}/index.m3u8`) works on any network if port 8554 is blocked.
- **Frame subsampling**: controlled by `FRAME_SKIP` (default 5 = process every 5th frame ≈ 6 FPS). Tune up/down based on GPU load.
- **CLIP checkpoint**: `ViT-B-32` / `openai` weights — **the backend must use the identical checkpoint** for NL search text encoding. Different weights = broken cosine similarity.
- **ANPR is best-effort**: plates are null when unreadable. Cross-camera re-ID via CLIP embeddings works regardless.
- **Watchlist alert**: plate match (exact string) checked before embedding match (cosine similarity ≥ 0.75). On match: row inserted into `alerts` + `pg_notify('setu_alerts', alert_id)` fires automatically via DB trigger.

## Database schema

See [`scripts/schema.sql`](scripts/schema.sql). Key tables:

| Table | Description |
|---|---|
| `cameras` | Sentinel sandbox cameras with lat/lng |
| `detections` | One row per detected object per frame; `embedding vector(512)` for HNSW search |
| `watchlist_entries` | Plates or reference embeddings to watch for |
| `alerts` | Pending/confirmed/dismissed watchlist matches |

## Integration with Model 1 (Sentinel CCTV Registry)

Model 1 provides the camera registry. In production, the ML worker queries
`GET /api/v1/cameras/{id}/streams` on Model 1's FastAPI for stream URLs.
For the hackathon build, we connect directly to the Sentinel sandbox
(`cctv.corp8.cloud`) which Model 1 also uses as its upstream source.

## Team handoff (for backend developer)

The backend FastAPI (Setu) needs to:
1. Connect to the same Postgres + pgvector database this worker writes to
2. Load `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')` for NL search text encoding
3. Listen for `pg_notify('setu_alerts', ...)` on a dedicated asyncpg connection and push to WebSocket
4. Serve `thumbnail_key` values as presigned MinIO URLs (never raw keys)
