import os
# import asyncio
from huggingface_hub import InferenceClient # type: ignore

from dotenv import load_dotenv # type: ignore
load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
token = os.getenv("HF_TOKEN", None)

client = InferenceClient(
    provider="featherless-ai",
    api_key=token
)

def main():
    completion = client.chat.completions.create(
        model=BASE_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Generate a short greeting message for a user named Anshul."
            }
        ],
        max_tokens=50
    )
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    main()
