# 🏛️ Setu Surveillance ML — Mega Architecture Document

This document is the definitive guide to the Setu ML Repository. It is designed for Infrastructure, DevOps, and Backend engineers to completely understand the internal workflows, scaling characteristics, data contracts, and deployment modes of the ML pipeline.

---

## 1. High-Level Design (HLD)

The Setu ML pipeline bridges the gap between raw, encrypted CCTV camera feeds and intelligent, semantic search. It ingests video, extracts rich metadata using an ensemble of AI models, and persists the data to a Vector Database (PostgreSQL + pgvector) and Object Storage (MinIO) for the Backend to consume.

```mermaid
graph TD
    subgraph "External Sandbox (cctv.corp8.cloud)"
        C[Cameras 1-30] -->|AES Encrypted HLS| HLS[HLS Server]
        Auth[Sentinel Auth API]
    end

    subgraph "Setu ML Infrastructure Layer"
        ML[ML Orchestrator]
        ML -->|Fetch Cookie| Auth
        ML -->|Download Segments| HLS
        
        subgraph "AI Inference Engine"
            YOLO[YOLOv8s: Object Detection]
            BYTE[ByteTrack: Object Tracking]
            CLIP[OpenCLIP ViT-B/32: Embeddings + NLP]
            OCR[Fast-Plate-OCR: License Plates]
        end
        
        ML --> YOLO
        YOLO --> BYTE
        BYTE --> CLIP
        CLIP --> OCR
    end

    subgraph "Data Persistence"
        PG[(PostgreSQL + pgvector)]
        MINIO[(MinIO Object Storage)]
    end

    ML -->|SQL Inserts + pg_notify| PG
    ML -->|Upload Crop JPEGs| MINIO

    subgraph "Backend / Frontend (Model 1 / Model 2)"
        API[FastAPI Backend]
        UI[React Frontend]
        PG -.->|WebSocket Triggers| API
        PG -.->|Cosine Similarity Search| API
        MINIO -.->|Image URLs| API
        API -.-> UI
    end
```

---

## 2. Low-Level Design (LLD): The Inference Pipeline

The core logic resides in `ml/pipeline.py` (`DetectionPipeline`). For every sampled frame, the system performs a complex sequence of inferences.

```mermaid
sequenceDiagram
    participant Stream as HLS Ingest (stream.py)
    participant Pipe as Pipeline (pipeline.py)
    participant YOLO as YOLOv8s (GPU)
    participant Track as ByteTrack (CPU)
    participant CLIP as OpenCLIP (GPU)
    participant OCR as PlateOCR (CPU/GPU)
    
    Stream->>Pipe: process_frame(frame_bgr)
    Pipe->>YOLO: Infer Bounding Boxes
    YOLO-->>Pipe: Boxes & Confidences
    
    Pipe->>Track: Update tracker (boxes.cpu())
    Track-->>Pipe: Track IDs mapped to Bboxes
    
    loop For each tracked vehicle
        Pipe->>Pipe: Crop frame to Bbox
        
        Pipe->>CLIP: encode_image(crop)
        CLIP-->>Pipe: 512-dim Vector Embedding
        
        Note over Pipe,CLIP: Zero-Shot Classification
        Pipe->>CLIP: Dot product with Pre-encoded Text Labels
        CLIP-->>Pipe: Argmax -> Class (e.g., 'auto_rickshaw')
        Pipe->>CLIP: Dot product with Pre-encoded Color Labels
        CLIP-->>Pipe: Argmax -> Dominant Color (e.g., 'red')
        
        Pipe->>OCR: Extract text from Crop
        OCR-->>Pipe: License Plate String
    end
    
    Pipe-->>Stream: List[Detection objects]
```

### Key AI Characteristics:
* **Zero-Shot NLP via CLIP:** We do not rely on YOLO's standard COCO classes (which lack Indian vehicles like Auto Rickshaws) or rudimentary HSV color buckets (which suffer from glare). We use CLIP to encode text prompts (`"a photo of an auto_rickshaw"`, `"a photo of a red vehicle"`) and take the mathematical dot-product against the cropped image embedding to classify vehicles.
* **Tracking:** ByteTrack maintains temporal state across frames. It prevents the system from triggering 30 alerts for the same stopped car across 30 frames.

---

## 3. Low-Level Design (LLD): Video Ingestion & Decryption

The network ingest layer (`ml/stream.py`) handles the unreliable corporate sandbox network.

```mermaid
graph LR
    M[Fetch m3u8 Manifest] --> S[Resolve Time Window]
    S --> B[Segment Queue]
    
    subgraph "Double-Buffered Fetcher Thread"
        B -->|Fetch N| HTTP[HTTP GET .ts file]
        HTTP --> AES[AES-128 CBC Decrypt]
        AES --> Disk[Write to tmpfs]
    end
    
    Disk --> OpenCV[cv2.VideoCapture]
    OpenCV --> Ski[Frame Skip/Sample Logic]
    Ski --> Out[Yield Frame to Pipeline]
```

* **Decryption:** The `.m3u8` manifest contains an AES key URI. `stream.py` fetches the key, derives the IV, and decrypts the segment on the fly.
* **Timestamping:** `cv2.CAP_PROP_POS_MSEC` is read during decoding to generate the `pts_ms` field. This is critical for the frontend video player to seek to the exact millisecond a vehicle appeared.

---

## 4. Execution Modes: Live Worker vs. Batch Indexer

The ML repository operates in two fundamentally different modes, dictating how the infrastructure should deploy them.

### A. The Live Worker (`python -m ml.worker`)
* **Purpose:** Production, real-time 24/7 surveillance daemon.
* **Concurrency:** Spawns **30 simultaneous threads** (one per camera).
* **Sampling:** Continuous. Reads a steady N frames-per-second to maintain perfect ByteTrack state.
* **Anomaly Engine:** Active. Because tracking is continuous, `worker.py` monitors centroid drift. If a centroid moves `< 20px` over `30` consecutive processed frames (~5 seconds), it triggers a `stopped_vehicle` anomaly.
* **Watchlist Engine:** Active. Matches license plates against the DB and instantly inserts `alerts`.
* **Hardware:** Requires massive compute (Heavy GPU + High core count CPU) to sustain 30 simultaneous streams.

### B. The Batch Indexer (`python -m scripts.batch_index`)
* **Purpose:** One-off historical indexing to populate the Vector DB for search demos.
* **Concurrency:** Sequential. Processes Camera 1, then Camera 2, etc.
* **Sampling:** **Burst-Sampling**. To bypass network bottlenecks, it downloads exactly **100 evenly-spaced segments** (1 every ~7 minutes) across the 12-hour archive. Inside those 6-second segments, it processes at 2 FPS.
* **Anomaly Engine:** Disabled. Burst sampling destroys continuous tracking, making anomaly detection impossible here.
* **Hardware:** Runs cleanly on a single consumer GPU (e.g., RTX 3060) within 8 hours for all 30 cameras.

---

## 5. PostgreSQL Schema & API Contract

The `scripts/schema.sql` file defines the contract between the ML layer and the Backend API.

### Tables
1. **`cameras`**: Seeded initially. Contains `external_id` (e.g., `cam04`) and geo-coordinates.
2. **`detections`**: The massive fact table.
   * `embedding`: `vector(512)`. Search is performed via HNSW index: `ORDER BY embedding <=> query_vector ASC`.
   * `pts_ms`: Float. Use this to seek the frontend video player.
   * `thumbnail_key`: MinIO path (e.g., `cam04/2026/...jpg`).
3. **`watchlist_entries`**: User-defined license plates to watch for.
4. **`alerts`**: Fired by `worker.py`. 
   * `alert_type`: enum(`watchlist_match`, `stopped_vehicle`, `wrong_way`).
   * `watchlist_entry_id`: NULL if the alert was a rule-based anomaly (like a stopped vehicle).

### The Event Bus (`NOTIFY 'setu_alerts'`)
Every `INSERT` into the `alerts` table fires a PostgreSQL trigger.
* **Backend Responsibility:** The backend must establish an async Postgres connection, execute `LISTEN setu_alerts;`, and push WebSocket messages to the frontend whenever a payload (the Alert UUID) arrives.

---

## 6. Directory Map & Script Reference

An exhaustive list of every script in the repository.

### Core Modules (`ml/`)
* `auth.py`: Handles Sentinel sandbox login, cookie extraction, and HTTP retry logic.
* `config.py`: Environment variable loading (Pydantic).
* `db.py`: PostgreSQL connection pooling and insert statements.
* `models.py`: Lazy-loaded, singleton initializers for YOLO, CLIP, and OCR models to prevent memory leaks and VRAM fragmentation.
* `pipeline.py`: The `DetectionPipeline` class (Inference orchestration).
* `storage.py`: MinIO S3 operations.
* `stream.py`: `CameraStream` context manager, AES decryption, HLS handling, and burst sampling logic.
* `watchlist.py`: Caches watchlist entries and runs matching logic.
* `worker.py`: The multi-threaded live daemon.

### CLI Scripts (`scripts/`)
* `batch_index.py`: The historical backfill tool (burst-sampling).
* `reset.py`: DANGER! Truncates Postgres, drops schemas, wipes MinIO buckets clean.
* `schema.sql`: The raw SQL DDL used by `reset.py`.
* `seed.py`: Fetches the camera list from the Sentinel API and populates the `cameras` table. Must be run before anything else.
* `test_search.py`: Local CLI tool to test NLP searches. E.g., `python -m scripts.test_search "yellow auto rickshaw"`. Encodes the text to a vector and queries Postgres directly to print terminal results.

---

## 7. Infrastructure Requirements (For DevOps)

### Environment Variables Required
```env
DATABASE_URL=postgresql://setu:setu_dev_only@127.0.0.1:5433/setu
MINIO_URL=127.0.0.1:9000
MINIO_ACCESS_KEY=setu_minio
MINIO_SECRET_KEY=setu_minio_secret
SENTINEL_USERNAME=YOUR_USERNAME
SENTINEL_PASSWORD=YOUR_PASSWORD
```

### System Requirements
* **RAM:** 16GB Minimum (Model weights take ~4GB RAM, plus Python threading overhead).
* **VRAM:** 6GB+ NVidia GPU (RTX 3060 tested). PyTorch MUST be compiled with CUDA.
* **Network:** High bandwidth required for the Live Worker (pulling 30 concurrent 1080p HLS streams).
* **Disk:** Temporary storage required in `/tmp` (or memory) for AES-decrypted video chunks before they are flushed to RAM via OpenCV.
