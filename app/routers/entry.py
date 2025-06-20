from typing import Optional, List, Literal
from fastapi import APIRouter, HTTPException # type: ignore
from pydantic import BaseModel, model_validator # type: ignore
from app.storage import save_entry  # type: ignore
from app.hf_client import generate_entry # type: ignore

router = APIRouter()

# 10 psychology-backed journaling prompts
QUESTIONS = [
    "How am I feeling right now?",
    "What was the best part of my day?",
    "What obstacles did I face, and how did I respond (or how might I respond next time)?",
    "List three things I’m grateful for today, and why.",
    "What did I learn today—about myself, others, or the world?",
    "How did I take care of my needs (physical, mental, social) today?",
    "Who made a positive impact on my day (even in a small way)?",
    "What frustrated or stressed me today, and what helped me cope?",
    "What surprised me or felt unexpected, and how did it affect me?",
    "What’s one intention or hope I have for tomorrow?"
]

class EntryRequest(BaseModel):
    mode: Literal["manual", "ai"]
    user_id: str
    answers: Optional[List[str]] = None # for AI mode
    content: Optional[str] = None # for manual mode
    
class EntryResponse(BaseModel):
    entry_id: str
    text: str

@router.post("", response_model=EntryResponse)
async def create_entry(req: EntryRequest):
    # Manual mode: save directly
    if req.mode == "manual":
        if not req.content or not req.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty.")
        entry_text = req.content.strip()
        entry_id, text, *_ = save_entry(req.user_id, entry_text)
        return {"entry_id": entry_id, "text": entry_text}
    # AI mode: generate entry from answers
    if req.mode == "ai":
        if not req.answers or len(req.answers) != len(QUESTIONS):
            raise HTTPException(status_code=400, detail=f"AI mode requires exactly {len(QUESTIONS)} answers.")
        raw_block = "\n\n".join(f"{q}\n{a}" for q, a in zip(QUESTIONS, req.answers))

        try:
            ai_text = await generate_entry(raw_block, req.user_id)
        except Exception:
            ai_text = raw_block
        entry_id, text, *_ = save_entry(req.user_id, ai_text)
        return {"entry_id": entry_id, "text": text}

    raise HTTPException(status_code=400, detail="Invalid mode. Use 'manual' or 'ai'.")