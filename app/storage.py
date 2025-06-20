import os
import json
from datetime import datetime, timedelta

import faiss # type: ignore
import numpy as np
#from huggingface_hub import InferenceClient # type: ignore

# Root Data Directory
DATA_DIR = os.getenv("DATA_DIR", "./data")

# HF Token and Model
HF_TOKEN = os.getenv("HF_TOKEN")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Inference API
#embed_api = InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

# Milestones for badges
_BADGE_MILESTONES = {
    3: "3-day streak",
    7: "7-day streak",
    14: "14-day streak",
    30: "30-day streak"
}

# Path helper functions
def _user_base(user_id: str) -> str:
    return os.path.join(DATA_DIR, user_id)

def _entries_dir(user_id: str) -> str:
    return os.path.join(_user_base(user_id), "entries")

def _index_dir(user_id: str) -> str:
    return os.path.join(_user_base(user_id), "index")

def _meta_path(user_id: str) -> str:
    return os.path.join(_user_base(user_id), "meta.json")

def _index_path(user_id: str) -> str:
    return os.path.join(_index_dir(user_id), "faiss.index")

def _id_map_path(user_id: str) -> str:
    return os.path.join(_index_dir(user_id), "id_map.json")

# Directory creating
def _ensure_user_dirs(user_id: str):
    os.makedirs(_entries_dir(user_id), exist_ok=True)
    os.makedirs(_index_dir(user_id), exist_ok=True)

# Metadata management
def load_meta(user_id: str) -> dict:
    path = _meta_path(user_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {"entries": [], "streak": 0, "badges": []}

def save_meta(user_id: str, meta: dict):
    with open(_meta_path(user_id), 'w') as f:
        json.dump(meta, f, indent=2)

def _update_meta(user_id: str, entry_date: str):
    """
    Add a new entry date (ISO 'YYYY-MM-DD') and compute streak + badge.
    Returns: (streak, badge_awarded or None)
    """
    meta = load_meta(user_id)

    # Add unique, sorted
    if entry_date not in meta["entries"]:
        meta["entries"].append(entry_date)
        meta["entries"].sort()
    
    # Compute streak
    streak = 0
    d = datetime.fromisoformat(entry_date).date()
    while d.isoformat() in meta["entries"]:
        streak += 1
        d -= timedelta(days=1)
    meta["streak"] = streak

    # Check for badges
    badge = None
    if streak in _BADGE_MILESTONES:
        name = _BADGE_MILESTONES[streak]
        if name not in meta["badges"]:
            meta["badges"].append(name)
            badge = name
    
    save_meta(user_id, meta)
    return streak, badge

# Faiss index management
def _load_or_create_index(user_id: str, dim: int) -> faiss.Index:
    """Load existing FAISS index or return a new FlatL2 index of dimension `dim`."""
    path = _index_path(user_id)
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexFlatL2(dim)

def _save_index(user_id: str, index: faiss.Index):
    faiss.write_index(index, _index_path(user_id))

def _load_id_map(user_id: str, index: faiss.Index):
    """Map FAISS internal IDs to our entry_id strings."""
    path = _id_map_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _save_id_map(user_id: str, id_map: dict):
    """Save mapping of FAISS internal IDs to entry_id strings."""
    with open(_id_map_path(user_id), "w") as f:
        json.dump(id_map, f, indent=2)

# Embedding management
def _embed_text(text: str) -> np.ndarray:
    """
    Call HF InferenceApi to get a 1D float32 embedding for `text`.
    """
    # try:
    #     result = embed_api.feature_extraction(inputs=text)
    #     vec = np.array(result[0], dtype="float32")
    # except Exception:
    #     vec = np.zeros(384, dtype="float32")  # Fallback to zero vector if embedding fails
    # return vec
    # Stubbed version
    return np.zeros(384, dtype="float32")  # Replace with actual embedding logic

# Entry ID Generation
def _generate_entry_id() -> str:
    """
    Unique ID for each entry: UTC timestamp in ISO-ish form.
    e.g. '20250618T154312Z'
    """
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# Public Save Function
def save_entry(user_id: str, content: str):
    """
    1) Ensure directories exist
    2) Write `content` to a new .txt file named by entry_id
    3) Update streak & badges based on entry date
    4) Embed + add to FAISS index + update ID map
    Returns:
      entry_id (str), content (str), streak (int), badge_awarded (str|None)
    """
    # Ensure user directories exist
    _ensure_user_dirs(user_id)

    # Write file
    entry_id = _generate_entry_id()
    file_path = os.path.join(_entries_dir(user_id), f"{entry_id}.txt")
    with open(file_path, 'w') as f:
        f.write(content)
    
    # Update metadata
    date_iso = entry_id[:8]  # Extract date from ID
    date_iso = f"{date_iso[:4]}-{date_iso[4:6]}-{date_iso[6:]}"  # Convert to YYYY-MM-DD
    streak, badge = _update_meta(user_id, date_iso)

    # RAG indexing
    embedding = _embed_text(content) # 1D (dim,)
    dim = embedding.shape[0]
    index = _load_or_create_index(user_id, dim)
    id_map = _load_id_map(user_id, index)

    new_idx = index.ntotal 
    index.add(embedding.reshape(1, dim)) # add new vector
    id_map[str(new_idx)] = entry_id  # Map FAISS ID to our entry_id

    _save_index(user_id, index)
    _save_id_map(user_id, id_map)

    return entry_id, content, streak, badge
