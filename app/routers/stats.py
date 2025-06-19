from fastapi import APIRouter # type: ignore
from app.storage import load_meta

router = APIRouter()

@router.get(
    "/{user_id}",
    summary="Get journaling statistics for a user",
    response_model=dict,
)

def stats(user_id: str):
    """
    Returns total number of entries, current streak, and badges earned.
    """
    meta = load_meta(user_id)
    return {
        "total_entries": len(meta["entries"]),
        "streak": meta["streak"],
        "badges": meta["badges"],
    } 