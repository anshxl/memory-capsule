import os
from huggingface_hub import InferenceClient # type: ignore

from dotenv import load_dotenv # type: ignore
load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
token = os.getenv("HF_TOKEN", None)

client = InferenceClient(model=EMBED_MODEL, token=token)

def main():
    example = "This is a test sentence for embeddings."
    result = client.feature_extraction(text=example)
    vec = result
    print(f"Got embedding of length {len(vec)} ({vec[:5]}...)")

if __name__ == "__main__":
    main()