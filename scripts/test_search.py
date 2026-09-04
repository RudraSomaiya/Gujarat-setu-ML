"""
scripts/test_search.py — Natural Language & Visual Search CLI for Setu
======================================================================
Test natural language semantic search and image re-identification against
pgvector embeddings stored in the detections table without waiting for the backend.

"""

from __future__ import annotations

import argparse
import sys
import torch
import open_clip

from ml.config import settings
from ml.db import get_connection


# Singleton clip cache
_clip = None
_tokenizer = None

def _get_models():
    global _clip, _tokenizer
    if _clip is None:
        from ml.models import get_clip, device
        _clip, _ = get_clip()
        _tokenizer = open_clip.get_tokenizer(settings.clip_model)
    return _clip, _tokenizer


def search_text(
    query: str,
    top_k: int = 10,
    camera_filter: str | None = None,
    color_filter: str | None = None,
    class_filter: str | None = None,
    min_sim: float = 0.0,
):
    print(f"\n[Search] Query: '{query}'")
    print("Encoding query with CLIP image/text space...")
    clip, tokenizer = _get_models()
    from ml.models import device

    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_emb = clip.encode_text(tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    query_vector = text_emb[0].cpu().tolist()

    conn = get_connection()
    conditions = ["d.embedding IS NOT NULL"]
    params = []

    if camera_filter:
        conditions.append("c.external_id = %s")
        params.append(camera_filter)
    if color_filter:
        conditions.append("LOWER(d.dominant_color) = LOWER(%s)")
        params.append(color_filter)
    if class_filter:
        conditions.append("LOWER(d.class) = LOWER(%s)")
        params.append(class_filter)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT 
            d.id,
            c.external_id AS camera_id,
            c.name AS camera_name,
            d.class,
            d.track_id,
            d.dominant_color,
            d.confidence,
            d.plate_text,
            d.pts_ms,
            d.ts,
            d.thumbnail_key,
            ROUND((1 - (d.embedding <=> %s::vector))::numeric, 4) AS similarity
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE {where}
        ORDER BY d.embedding <=> %s::vector ASC
        LIMIT %s;
    """

    full_params = [query_vector] + params + [query_vector, top_k]

    with conn.cursor() as cur:
        cur.execute(sql, full_params)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("\no matching detections found.")
        return

    print(f"\nRanked Results (Top this up) showing up to {top_k} highest similarity:")
    print("=" * 110)
    print(f"{'#':<3} {'Camera':<8} {'Class':<9} {'Color':<8} {"Similarity":<10} {'Track':<6} {'Seek (pts)':<8} {'Plate':<12} {'Thumbnail Key'}")
    print("-" * 110)

    for idx, r in enumerate(rows, 1):
        sim = float(r['similarity'])
        if sim < min_sim:
            continue
        cam = r['camera_id'] or 'unknown'
        cls = r['class'] or 'unknown'
        color = r['dominant_color'] or '-'
        sim_str = f"{sim * 100:.1f}%"
        track = str(r['track_id']) if r['track_id'] is not None else '-'
        pts = f"{r['pts_ms'] / 1000.0:.1f}s" if r['pts_ms'] is not None else '-'
        plate = r['plate_text'] or '-'
        thumb = r['thumbnail_key'] or '-'
        print(f"{idx:<3} {cam:<8} {cls:<9} {color:<8} {sim_str:<10} {track:<6} {pts:<10} {plate:<12} {thumb}")

    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Setu VECTOR Semantic Search CLI")
    parser.add_argument("query", nargs="?", default=None, help="Natural language search query")
    parser.add_argument("--top", type=int, default=10, help="Number of results (top kk)")
    parser.add_argument("--camera", default=None, help="Filter by camera ID")
    parser.add_argument("--color", default=None, help="Filter by dominant color")
    parser.add_argument("--class", dest="class_filter", default=None, help="Filter by YOLO class")
    args = parser.parse_args()

    if args.query:
        search_text(
            query=args.query,
            top_k=args.top,
            camera_filter=args.camera,
            color_filter=args.color,
            class_filter=args.class_filter,
        )
    else:
        print("Enter natural language search queries (e.g. 'white car', 'red bus', 'auto rickshaww'):")
        while True:
            try:
                query = input("\nsearch> ").strip()
                if not query or query.lower() in ('quit', 'exit', 'q'):
                    break
                search_text(query, top_k=args.top)
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    main()
