import os
import json
from fastapi import APIRouter, HTTPException, Query # type: ignore
from typing import List
import numpy as np
import faiss # type: ignore

from app.storage import (
    _embed_text,
    _index_path,
    _id_map_path,
    _entries_dir,
)

router = APIRouter()

@router.get(
    "/{user_id}",
    response_model=List[dict],
    summary="Retrive poast entries related to query",
)

def flashback(
    user_id: str,
    q: str = Query(..., description="Flashback query string"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """
    Returns up to k of the user’s past entries whose embeddings are closest
    to the query string `q`.
    """
    # ed = _entries_dir(user_id)
    # if not os.path.exists(ed):
    #     return []
    # files = sorted(os.listdir(ed))[:k]  # Get the first k files
    # results = []
    # for fn in files:
    #     path = os.path.join(ed, fn)
    #     with open(path, "r") as f:
    #         content = f.read()
    #     entry_id = fn.rsplit(".", 1)[0]  # Extract entry ID from filename
    #     results.append({"entry_id": entry_id, "content": content})
    index_file = _index_path(user_id)
    id_map_file = _id_map_path(user_id)
    entries_dir = _entries_dir(user_id)

    if not os.path.exists(index_file) or not os.path.exists(id_map_file):
        raise HTTPException(status_code=404, detail="No entries found for this user")
    
    # Load FAISS mapping
    with open(id_map_file, "r") as f:
        id_map = json.load(f)
    
    # Load FAISS index
    index = faiss.read_index(index_file)

    # Embed the query
    emb = _embed_text(q).astype("float32")
    D, I = index.search(np.array([emb]), k)

    # Retrieve the actual entries
    results = []
    for dist, idx in zip(D[0], I[0]):
        entry_id = id_map.get(str(int(idx)))
        if not entry_id:
            continue
        path = os.path.join(entries_dir, f"{entry_id}.txt")
        try:
            content = open(path, "r").read()
        except FileNotFoundError:
            content = ""
        results.append({
            "entry_id": entry_id,
            "content":  content,
            "score":    float(dist)
        })
    return results