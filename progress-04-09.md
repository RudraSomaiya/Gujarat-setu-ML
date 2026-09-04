# Setu ML Pipeline — Session Discussion Log
**Role:** ML Engineer (Rudra)  
**Project:** Gujarat Police Innovation Challenge 2026 — Setu surveillance intelligence platform  
**Hackathon timeline:** 10 days

---

## 1. Project Context

Setu is a federation and AI intelligence layer sitting on top of Gujarat's existing multi-vendor CCTV infrastructure. It does not replace any hardware. It ingests camera feeds, runs inference, and writes structured detection events to a PostgreSQL database.

The full system involves four roles: infra/devops, backend, frontend, and ML engineer. The ML engineer owns the `ml/` folder and is the only one writing to the `detections` table and MinIO.

The tech stack is locked: YOLOv8, ByteTrack, a pretrained plate detector + OCR for ANPR, and OpenCLIP ViT-B/32 for visual and text embeddings. Output goes to PostgreSQL (with pgvector extension) via SQLAlchemy.

---

## 2. Current Progress (as of session start)

The ML engineer has built and tested the following:

- **Camera feed ingestion:** 30 camera feeds fetched from the hackathon host's website. These are 12-hour pre-recorded videos. HLS/TCP streaming was attempted; naive fetching caused grey smearing artifacts in video frames. This was identified and fixed.
- **PostgreSQL + MinIO:** Both running inside Docker containers.
- **YOLOv8s:** Detection and crop pipeline working. Detections are being written to the DB and thumbnails stored in MinIO. Currently using `yolo.predict()`, not `yolo.track()` — see ByteTrack section below.
- **OpenCLIP embeddings:** ✅ Done. For every vehicle YOLO detects, the crop is run through OpenAI CLIP (ViT-B-32), the vector is L2-normalized, and the embedding is written to the `detections` table. Fully wired and ready for vector search.
- **ByteTrack:** ❌ Not working. `track_id` is currently always `None`. See section 10 for full details.
- **ANPR:** A plate OCR model (`fast-plate-ocr` or similar) is running on vehicle crops. Plate strings are being written to the DB where detected. Quality is poor due to 480p resolution — deprioritized (see section 3).
- **Testing scripts:** Multiple scripts created — for streaming, for live YOLO visualization, and for the final integrated ML worker.
- **Plate alert/watchlist:** Implemented. A seeded plate in `watchlist_entries` triggers a `pending` alert on match.

---

## 3. ANPR Problem — Decision to Deprioritize

### What was observed

Several processed plate crops were examined (upscaled and sharpened before OCR). Even under best-case conditions (good angle, better lighting), OCR output was largely unreliable. A few partial reads were possible (e.g., partial `GJ11CH2`, `GJ03FK3132`), but consistent full plate reads were not achievable.

### Root cause identified

The camera feeds are 480p. A license plate at any real-world distance on a 480p feed occupies approximately 15–30 pixels of width in the raw frame. Compression artifacts (H.264/HLS) further destroy fine detail. Upscaling and sharpening reconstruct noise, not recoverable plate characters. Even regex post-processing on OCR output produces garbage strings.

### Decision made

**ANPR is deprioritized as a primary feature.** The plate alert watchlist code stays in the pipeline because it will fire correctly on the occasional high-confidence readable frame. However, no more engineering effort will go into improving ANPR accuracy on this camera hardware. The correct architectural framing is: ANPR is available when camera resolution supports it; the visual embedding fallback handles cases where it does not. This is explicitly how the architecture was designed.

### Demo framing

ANPR is presented as a hardware-dependent feature. At 480p, visual re-identification via embeddings is the primary matching mechanism. At 2MP+ (the real installed base being upgraded), ANPR works without changing any pipeline code.

---

## 4. YOLOv8 COCO Classes and Indian Vehicles

### Question raised

COCO (the dataset YOLOv8 was trained on) does not include auto-rickshaws as a class. Does this break the "search for yellow auto-rickshaw" use case?

### Resolution

The detection model (YOLOv8) and the embedding model (OpenCLIP) have separate jobs:

- **YOLOv8's job** is to find bounding boxes around vehicle-shaped regions. The class label is used only for coarse filtering. Whether an auto-rickshaw is labelled `car` or `motorcycle` by YOLO does not affect downstream embedding quality.
- **CLIP's job** is to encode the visual content of the crop into a semantic embedding space. CLIP was trained on internet-scale data including Indian street imagery. Its text encoder understands "yellow auto-rickshaw" as a concept, and its image encoder produces embeddings for auto-rickshaw crops that are geometrically close to that text concept in embedding space.

The NL search pipeline therefore works for Indian-specific vehicle types even without COCO having those classes. YOLO finds things; CLIP understands them. These are complementary.

---

## 5. Cross-Camera Tracking — Decision to Deprioritize

### Situation

The 30 camera feeds are provided by the hackathon host. All teams have access to the same feeds. The cameras are not at the same physical location, do not share synchronized timestamps in any known way, and no GPS/GIS location metadata is available for any of them.

### Problem with cross-camera tracking

Cross-camera tracking (the animated GIS trajectory replay feature from the original architecture) requires knowing the physical coordinates of each camera to plot a route on a map. Without location metadata, there is no map trajectory to animate — only a list of camera IDs which is not visually compelling.

Embedding-based cross-camera matching (finding the same-looking vehicle across feeds) is still technically possible but without geography becomes a "timeline of sightings" rather than a map route.

### Decision made

**Cross-camera GIS trajectory tracking is not a viable primary feature given the dataset constraints.** It may be attempted as a secondary feature if time permits, presented as a "timeline" view rather than a map view. It is not a priority.

---

## 6. Feature Pivot — Temporal NL Search (Primary Demo Feature)

### The new framing

The actual problem being solved is not "find a car across cameras." It is: **given 30 cameras × 12 hours = 360 hours of footage, find every moment where a specific type of vehicle or person appears, instantly.**

This is directly what the hackathon context enables: the host has given all teams the same 30 feeds. The teams who win will be the ones who make those feeds searchable in ways that save investigator time. Manual review of 360 hours of footage is the baseline. The Setu pipeline reduces that to a semantic query returning ranked thumbnails with timestamps.

### How it works

1. The ML worker processes footage and writes detection rows to Postgres with timestamps.
2. Every detection has a CLIP image embedding stored in the `vector(512)` column.
3. An operator types a natural language query: "red SUV," "person in blue shirt," "yellow auto-rickshaw."
4. The backend encodes the query text with the same CLIP text encoder (ViT-B/32, `openai` weights — must match exactly).
5. A pgvector cosine similarity search runs against all stored embeddings across all 30 cameras.
6. Results are returned ranked by similarity, each with: camera ID, timestamp, thumbnail from MinIO.
7. The operator jumps directly to that moment in the footage.

### The timestamp-to-video-seek requirement

For this feature to be fully useful in the demo, the timestamp stored in the detection row needs to map to a seekable offset in the video player. If video start time and frame rate are known, the offset in seconds can be derived. Most HLS players support time-based seeking via fragment or query parameter. An operator clicking a result should jump directly to that 10-second window in the camera feed.

This detail has not been implemented yet and is identified as a required piece for the feature to feel complete.

### Why this is the right pivot

- Works entirely within what has already been built (YOLO crops → CLIP embeddings → Postgres)
- Requires no location metadata
- Directly addresses the manual-review bottleneck stated in the problem statement
- Produces a clear, demonstrable, time-saving result
- Differentiates from teams who focus on ANPR or map-based features

---

## 7. Compute Constraints — Batch Processing Strategy

### Problem

Running CLIP embeddings across 30 cameras × 12 hours of footage on a single laptop is not feasible if done naively (every frame, in real-time). The question was whether brute force is required or if there is a smarter approach.

### Key constraint quantified

At 15 FPS and 12 hours per camera, one camera produces approximately 648,000 frames. Across 30 cameras, that is approximately 19.4 million frames. Running YOLO + CLIP on every frame on CPU would take multiple days.

### Strategy agreed upon

Two optimizations reduce this to an overnight job:

**Optimization 1 — Frame sampling.** A vehicle moving through a camera's field of view at normal speed is visible for 3–10 seconds. Sampling one frame every 5 seconds captures every vehicle at least once. This reduces the frame count by approximately 75× (from ~19M to ~260K frames across all cameras).

**Optimization 2 — YOLO gates CLIP.** YOLO runs on every sampled frame (fast, CPU-friendly). CLIP only runs when YOLO detects a vehicle. On quiet cameras (night, low-traffic periods), CLIP may not run at all. This means the expensive model only fires when there is something meaningful to encode.

### Additional efficiency notes

- CLIP crops should be batched (multiple crops in a single forward pass) rather than run one at a time.
- GPU acceleration (NVIDIA CUDA, Apple MPS) reduces total processing time significantly. MPS on Apple Silicon is available if the laptop has it.
- Not all 12 hours of every feed need to be processed. Daytime hours (roughly 6am–10pm) contain the majority of traffic events. Processing 16 hours instead of 12 across 30 cameras is a reasonable scope reduction.
- Cameras should be processed in priority order — feeds with better resolution, lighting, and traffic density first. A partial run covering the best 15 cameras is more useful for demo than a failed full run.

### Recommended approach

Run a batch offline indexing job overnight. Start it before sleeping. By morning, the `detections` table has embeddings across all (or most) cameras, and the NL search demo works without any live inference latency during judging.

---

## 8. Open Items / What Is Not Yet Built

Current status of all tracked items:

- **OpenCLIP embeddings:** ✅ Done. Writing normalized 512-dim vectors to `detections.embedding` for every vehicle detection.
- **CLIP checkpoint confirmed:** ViT-B-32 / openai. Backend must use the same — confirm this has been communicated to backend teammate.
- **ByteTrack:** ❌ Not working. See section 10 for full details and the known fix. This is the current highest priority task.
- **Timestamp-to-video-seek:** ❌ Not implemented. The mapping from a detection's `ts` value to a seekable offset in the HLS stream is a required piece for the temporal search feature to feel complete in the demo.
- **Vehicle color extraction:** ❌ Not implemented. Simple HSV-based dominant color bucketing on each detection crop, stored as a column in `detections`. Improves precision for structured queries like "red vehicle" or "white SUV." Low effort, high demo value.
- **Batch offline indexing script:** ❌ Not written. A single script to iterate all 30 camera video files with the frame-sampling + gated-CLIP strategy and populate the DB overnight. Required before embeddings are usable at demo scale.
- **Watchlist embedding check:** ❌ Status unconfirmed. Plate-based watchlist alert is implemented. The embedding-based check (for `person_embedding` type entries, cosine similarity above threshold) has not been confirmed as implemented.

---

## 10. ByteTrack — Bug, Current State, and Fix Approach

### What was attempted

`yolo.track()` with the built-in ByteTrack integration was the original approach. This is documented in the Ultralytics API as the standard way to run ByteTrack.

### Bug encountered

When `yolo.track()` is used across multiple cameras running in parallel threads, the tracker state leaks between cameras. This is a known bug in Ultralytics when the same model object is shared across threads — the internal tracker state is not isolated per-camera-session. The result is that track IDs from one camera bleed into another, producing nonsensical tracking.

### Current workaround

Fell back to `yolo.predict()`. Detection works correctly. Tracking is completely disabled. `track_id` is `None` for every detection row in the database.

### Why this matters

Two downstream features depend on `track_id`:

1. **Cross-camera timeline grouping** — identifying the same vehicle across detections from the same camera requires stable per-camera track IDs.
2. **Best-of-N ANPR strategy** — a new OCR approach being considered where, instead of attempting OCR on every frame, you wait for N frames of the same tracked vehicle (identified by consistent `track_id`) and pick the single best-quality OCR read from those N frames. Without tracking, there is no way to know which frames belong to the same vehicle, so this strategy cannot be implemented.

### Agreed fix approach

Decouple ByteTrack completely from the YOLO model. Continue using `yolo.predict()` for detection. After YOLO returns bounding boxes for a frame, feed those boxes into a **separate, isolated ByteTracker instance maintained per camera**. Each camera gets its own tracker object that is never shared with any other camera thread. The `supervision` library provides a straightforward ByteTrack implementation that can be instantiated this way.

The structure is a dictionary keyed by camera ID, where each value is an independent tracker instance. YOLO detections for a given camera are passed only to that camera's tracker, which returns stable per-camera track IDs without cross-contamination.

### Status

Fix is designed and agreed upon. Not yet implemented. This is the current highest priority task.

---

## 9. Key Decisions Summary

| Decision | Outcome |
|---|---|
| ANPR accuracy improvement | Deprioritized. Keep pipeline, raise confidence threshold, no more effort on OCR accuracy. |
| Cross-camera GIS trajectory | Deprioritized. No location metadata available. May revisit as a timeline-only view. |
| Primary demo feature | Temporal NL semantic search across 360 hours of footage via CLIP embeddings. |
| CLIP model checkpoint | ✅ Confirmed: ViT-B-32 / openai pretrained weights. Must match across ML worker and backend. |
| CLIP embeddings | ✅ Done and writing to DB. |
| ByteTrack | ❌ Broken. `yolo.track()` multi-thread bug. Fix: isolated per-camera tracker instances via `supervision` library. Next priority. |
| Frame processing strategy | 1 frame per 5 seconds sampling. YOLO gates CLIP. Batched CLIP encoding. Run overnight. |
| Dataset | 30 camera feeds provided by hackathon host (12 hours each, 480p, HLS). Same for all teams. |
