import os
import argparse
import tempfile
from datasets import Dataset # type: ignore
from huggingface_hub import HfApi, Repository # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict # type: ignore

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
HF_ORG = os.getenv("HF_ORG")
BASE_MODEL = os.getenv("BASE_MODEL")

def parge_args():
    parser = argparse.ArgumentParser("Fine-tune a LoRA adapter on user voice samples")
    parser.add_argument("--user-id", required=True, help="User Identifier")
    parser.add_argument("--samples-file", required=True, 
                        help="Path to a newline-delimited text file with user samples")
    parser.add_argument("--repo-id", required=False, 
                        help="Full HF repo (org/adapter-repo). If omitted, uses HF_ORG/memory-capsule-adapter-{user_id}")
    parser.add_argument("--output-dir", default="adapter_output",
                        help="Local dir to store adapter weights before push")
    
def main():
    args = parge_args()
    user_id = args.user_id
    samples_file = args.samples_file   
    repo_id = args.repo_id or f"{HF_ORG}/memory-capsule-adapter-{user_id}"

    api = HfApi(token=HF_TOKEN)
    # Create adapter repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="adapter", exist_ok=True)
    except Exception as e:
        print(f"!! Repo creation warning: {e}")

    # Clone the repo to local
    tmp = tempfile.mkdtemp()
    repo = Repository(local_dir=tmp, 
                      clone_from=repo_id, 
                      token=HF_TOKEN, 
                      repo_type="adapter")

    # Load dataset
    with open(samples_file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    ds = Dataset.from_dict({"text": texts})

    # Load tokenizer and model onto GPU if available
    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # Set up LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)

    # TrainingArguments-push to hub when done
    training_args = TrainingArguments(
        output_dir=tmp,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=10,
        fp16=(device=="cuda"),
        push_to_hub=True,
        hub_model_id=repo_id,
        hub_token=HF_TOKEN,
    )

    # Trainer setup
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub()

    # Save the adapter weights locally
    adapter_state = get_peft_model_state_dict(peft_model)
    save_path = os.path.join(args.output_dir, "pytorch_model.bin")
    os.makedirs(args.output_dir, exist_ok=True)
    import torch # type: ignore
    torch.save(adapter_state, save_path)
    print(f"Adapter weights saved to {save_path}")

if __name__ == "__main__":
    main()