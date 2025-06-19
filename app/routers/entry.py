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
    
    @model_validator(mode='after')
    def check_required_fields(cls, values):
        mode = values.get('mode')
        answers = values.get('answers')
        content = values.get('content')

        if mode == "ai":
            if not answers or len(answers) != len(QUESTIONS):
                raise ValueError(f"AI mode requires exactly {len(QUESTIONS)} answers.")
        else:
            if not content or not content.strip():
                raise ValueError("Manual mode requires content to be provided.")
        return values

class EntryResponse(BaseModel):
    entry_id: str
    text: str

@router.post("", response_model=EntryResponse)
async def create_entry(req: EntryResponse):
    # Manual mode: save directly
    if req.mode == "manual":
        entry_text = req.content.strip()
        entry_id, _ = save_entry(req.user_id, entry_text)
        return {"entry_id": entry_id, "text": entry_text}
    # AI mode: generate entry from answers
    raw_block = "\n\n".join(f"{q}\n{a}" for q, a in zip(QUESTIONS, req.answers))

    ai_text = await generate_entry(raw_block, req.user_id)

    entry_id, _ = save_entry(req.user_id, ai_text)
    return {"entry_id": entry_id, "text": ai_text}