import os
import subprocess
from huggingface_hub import InferenceClient # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # type: ignore
import torch # type: ignore

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = os.getenv("BASE_MODEL")
HF_ORG = os.getenv("HF_ORG")
ADAPTERS_DIR = os.getenv("ADAPTERS_DIR")

# Setup Hugging Face API client
client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

async def generate_entry(user_id: str, raw_block: str) -> str:
    """
    1. Determine Adapter Repo for User
    2. Call HF Inference Endpoint with adapter parameter
    3. Return generated text
    """
    # return raw_block  # For now, just return the raw block
    messages = [
        {"role": "system", "content": "You are a warm, personal journalling assistant."},
        {"role": "user", "content": raw_block}
    ]
    completion = client.chat.completions.create(
        model=BASE_MODEL,
        messages=messages,
        max_tokens=500
    )
    return completion.choices[0].message.content

def train_adapter(user_id: str, samples: list[str]) -> str:
    """
    Stubbed function for training an adapter.
    """
    return f"{HF_ORG}/memory-capsule-adapter-{user_id}"

# def train_adapter(user_id: str, samples: list[str]) -> str:
#     """
#     1) Write `samples` to a temp file
#     2) Invoke train_adapter.py
#     Returns: the HF repo_id where the adapter was pushed
#     """
#     tmp_dir = os.path.join("data", user_id)
#     os.makedirs(tmp_dir, exist_ok=True)
#     samples_file = os.path.join(tmp_dir, "voice_samples.txt")

#     with open(samples_file, 'w') as f:
#         f.write("\n".join(samples))
    
#     repo_id = f"{HF_ORG}/memory-capsule-adapter-{user_id}"
#     cmd = [
#         "python", "train_adapter.py",
#         "--user-id", user_id,
#         "--samples-file", samples_file,
#         "--repo-id", repo_id,
#     ]

#     subprocess.run(cmd, check=True)
#     return repo_id