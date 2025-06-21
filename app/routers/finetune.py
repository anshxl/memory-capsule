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
def finetune(user_id: str, req: FinetuneRequest):
    """
    Placeholder endpoint for fine-tuning.
    """
    return {"message": "Fine-tuning coming soon!", "status": "unavailable"}