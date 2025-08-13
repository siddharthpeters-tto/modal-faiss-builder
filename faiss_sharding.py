from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import faiss

# -----------------------------
# Configuration
# -----------------------------
# Supabase Storage bucket name
BUCKET = "faiss"
# Optional folder prefix inside the bucket (e.g., "indexes"). Leave "" for bucket root.
OBJECT_DIR = ""
# Target shard size. Keep safely below hosted Supabase's per-object cap (~50 MB on Free/Pro).
TARGET_SHARD_MB = 30.0
# Manifest schema version (bump if you change structure)
MANIFEST_VERSION = 1


# -----------------------------
# Internal helpers
# -----------------------------

def _join_key(name: str) -> str:
    if not OBJECT_DIR:
        return name
    # Ensure single slash between
    if OBJECT_DIR.endswith("/"):
        return f"{OBJECT_DIR}{name}"
    return f"{OBJECT_DIR}/{name}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _serialize_size_mb(index: faiss.Index) -> float:
    return len(faiss.serialize_index(index)) / (1024 * 1024)


def _manifest_key(embed_type: str) -> str:
    return _join_key(f"clip_{embed_type}_manifest.json")


def _shard_key(embed_type: str, shard_id: int) -> str:
    return _join_key(f"clip_{embed_type}_shard_{shard_id:05d}.index")


# -----------------------------
# Build-time sharding state
# -----------------------------
class ShardState:
    """Keeps the current open shard per embedding type and the next shard id.

    dim_by_type: mapping like {"color": 512, "structure": 512, ...}
    """

    def __init__(self, dim_by_type: Dict[str, int]):
        self.dim_by_type = dict(dim_by_type)
        self.current_ix: Dict[str, faiss.Index] = {}
        self.next_shard_id: Dict[str, int] = {}

    def ensure_open(self, embed_type: str) -> None:
        if embed_type not in self.current_ix:
            d = self.dim_by_type[embed_type]
            self.current_ix[embed_type] = faiss.IndexFlatL2(d)
        if embed_type not in self.next_shard_id:
            self.next_shard_id[embed_type] = 0


# -----------------------------
# Manifest read/write
# -----------------------------

def load_manifest(supabase, embed_type: str) -> Dict:
    """Download manifest JSON for an embedding type. Returns empty manifest if missing."""
    path = _manifest_key(embed_type)
    try:
        raw = supabase.storage.from_(BUCKET).download(path)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"version": MANIFEST_VERSION, "embed_type": embed_type, "shards": []}


def save_manifest(supabase, manifest: Dict) -> None:
    path = _manifest_key(manifest["embed_type"])
    supabase.storage.from_(BUCKET).upload(
        path=path,
        file=json.dumps(manifest).encode("utf-8"),
        file_options={"contentType": "application/json", "upsert": "true"},
    )


# -----------------------------
# Rotation & upload during build
# -----------------------------

def _upload_shard(supabase, embed_type: str, shard_id: int, ix: faiss.Index) -> float:
    # Serialize FAISS index to bytes
    data_bytes = faiss.serialize_index(ix)
    path = _shard_key(embed_type, shard_id)
    supabase.storage.from_(BUCKET).upload(
        path=path,
        file=bytes(data_bytes),
        file_options={"contentType": "application/octet-stream", "upsert": "true"},
    )
    return len(data_bytes) / (1024 * 1024)



def _append_manifest_entry(supabase, embed_type: str, shard_id: int, ix: faiss.Index, approx_mb: float) -> None:
    manifest = load_manifest(supabase, embed_type)
    manifest["version"] = MANIFEST_VERSION
    manifest.setdefault("embed_type", embed_type)
    manifest.setdefault("shards", [])
    manifest["shards"].append(
        {
            "shard_id": int(shard_id),
            "file": _shard_key(embed_type, shard_id),  # full path within bucket
            "count": int(ix.ntotal),
            "created_at": _now_iso(),
            "approx_mb": round(float(approx_mb), 2),
        }
    )
    save_manifest(supabase, manifest)


def maybe_rotate_and_upload_shard(
    supabase,
    embed_type: str,
    shard_state: ShardState,
    target_mb: float = TARGET_SHARD_MB,
) -> None:
    """If the current shard exceeds target_mb, upload it and start a new one."""
    shard_state.ensure_open(embed_type)
    ix = shard_state.current_ix[embed_type]
    if ix.ntotal == 0:
        return
    size_mb = _serialize_size_mb(ix)
    if size_mb < target_mb:
        return
    shard_id = shard_state.next_shard_id[embed_type]
    approx_mb = _upload_shard(supabase, embed_type, shard_id, ix)
    _append_manifest_entry(supabase, embed_type, shard_id, ix, approx_mb)
    # rotate to fresh empty shard
    d = shard_state.dim_by_type[embed_type]
    shard_state.current_ix[embed_type] = faiss.IndexFlatL2(d)
    shard_state.next_shard_id[embed_type] = shard_id + 1


def flush_open_shard(supabase, embed_type: str, shard_state: ShardState) -> None:
    """Upload the current open shard even if below target size. Call at end of build."""
    shard_state.ensure_open(embed_type)
    ix = shard_state.current_ix[embed_type]
    if ix.ntotal == 0:
        return
    shard_id = shard_state.next_shard_id[embed_type]
    approx_mb = _upload_shard(supabase, embed_type, shard_id, ix)
    _append_manifest_entry(supabase, embed_type, shard_id, ix, approx_mb)
    # prepare next id for any future appends
    d = shard_state.dim_by_type[embed_type]
    shard_state.current_ix[embed_type] = faiss.IndexFlatL2(d)
    shard_state.next_shard_id[embed_type] = shard_id + 1


# -----------------------------
# Query-time loader (aggregate shards)
# -----------------------------

def load_sharded_index(
    supabase,
    embed_type: str,
    expected_dim: Optional[int] = None,
    threaded: bool = True,
) -> faiss.Index:
    """Load all shards for an embedding type into a FAISS IndexShards aggregator.

    If no shards exist, returns an empty IndexFlatL2 of expected_dim (or 512).
    """
    manifest = load_manifest(supabase, embed_type)
    shards_meta: List[dict] = manifest.get("shards", [])
    if not shards_meta:
        d = int(expected_dim or 512)
        return faiss.IndexFlatL2(d)

    # Load all shard indexes
    sub_indexes: List[faiss.Index] = []
    d_first: Optional[int] = None
    for s in shards_meta:
        path = s["file"]  # full path within bucket
        blob = supabase.storage.from_(BUCKET).download(path)
        sub = faiss.deserialize_index(blob)
        if d_first is None:
            d_first = int(sub.d)
        sub_indexes.append(sub)

    d = int(d_first or expected_dim or 512)

    # Aggregate shards. Some FAISS builds expect extra constructor params; try safest form.
    try:
        agg = faiss.IndexShards(d, threaded)
    except TypeError:
        # fallback for versions requiring (d, threaded, own_fields)
        agg = faiss.IndexShards(d, threaded, False)

    for sub in sub_indexes:
        agg.add_shard(sub)
    return agg
