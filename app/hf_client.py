import os
from huggingface_hub import InferenceClient, hf_api, HfApi # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # type: ignore
import torch # type: ignore

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL")
HF_ORG = os.getenv("HF_ORG")
ADAPTERS_DIR = os.getenv("ADAPTERS_DIR")

# Setup Hugging Face API client
api = HfApi(token=HF_TOKEN)
infer = InferenceClient(model=BASE_MODEL, token=HF_TOKEN)

async def generate_entry(user_id: str, raw_block: str) -> str:
    """
    1. Determine Adapter Repo for User
    2. Call HF Inference Endpoint with adapter parameter
    3. Return generated text
    """
    repo_id = f"{HF_ORG}/memory-capsule-adapter-{user_id}"
    prompt = (
        "You are my personal journalling assistant. "
        "Do NOT include any headings or commentary - only output the text. \n\n"
        "1) Write a 2 sentence highlight in the user's voice.\n"
        "2) Then compose a cohesive 300-350 word journal entry in their voice. \n\n"
        + raw_block
    )
    params = {"max_new_tokens": 600, "adapter": repo_id}
    resp = await infer.text_generation(prompt, params=params)
    return resp.generated_text

def train_adapter(user_id: str, samples: list[str]) -> str:
    """
    1) Create or clone HF adapter repo
    2) Fine-tune LoRA adapters on `samples` locally
    3) Push adapter weights back to HF under repo_id
    """
    from datasets import Dataset # type: ignore
    dataset = Dataset.from_dict({"text": samples})
    trained_repo = push_lora_adapter(BASE_MODEL, dataset, user_id)
    return trained_repo
