from fastapi import APIRouter, HTTPException #type: ignore
from pydantic import BaseModel #type: ignore
from app.hf_client import train_adapter

router = APIRouter()

class FinetuneRequest(BaseModel):
    samples: list[str]

class FinetuneResponse(BaseModel):
    adapter_repo: str
    status: str

@router.post("/{user_id}", response_model=FinetuneResponse)
async def finetune(user_id: str, req: FinetuneRequest):
    """
    Kick off LoRA fine-tuning for a given user_id.
    Returns the HF adapter repo and a status message.
    """
    if not req.samples or len(req.samples) < 10:
        raise HTTPException(status_code=400, detail="At least 10 samples are required for fine-tuning.")
    
    try:
        repo_id = train_adapter(user_id, req.samples)
    except Exception as e:
        raise HTTPException(500, f"Adapter training failed: {e}")
    
    return {"adapter_repo": repo_id, "status": "completed"}
