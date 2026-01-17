"""Modal launcher for continuing QLoRA fine-tuning with W&B.

This script loads a previously trained adapter and continues training
on new/additional data with Weights & Biases logging.

Usage:
    modal run modal_continue_train.py
    modal run modal_continue_train.py --max-steps 200
"""

from pathlib import Path

import modal

# Create Modal app
app = modal.App("qwen25-7b-tone-lora-continue")

# Use the existing adapter volume
volume = modal.Volume.from_name("qwen-tone-adapter", create_if_missing=False)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0,<2.4.0",
        "transformers>=4.41.0,<4.45.0",
        "datasets>=2.14.0",
        "peft>=0.7.0,<0.12.0",
        "trl>=0.8.0,<0.9.0",
        "accelerate>=0.25.0,<0.34.0",
        "bitsandbytes>=0.41.0,<0.44.0",
        "scipy",
        "sentencepiece",
        "rich",
        "wandb",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# HuggingFace cache volume
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=60 * 60 * 2,  # 2 hour timeout
    image=image,
    volumes={
        "/out": volume,
        "/cache": hf_cache,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def continue_training(
    data_content: str, max_steps: int = 200, learning_rate: float = 1e-4
):
    """Continue training from existing adapter with W&B logging.

    Args:
        data_content: The training data as a string (JSONL format)
        max_steps: Number of additional training steps
        learning_rate: Learning rate (lower for continued training)
    """
    import json
    import os
    import subprocess
    from pathlib import Path

    # Write training data to file
    data_path = "/tmp/tone_sft.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(data_content)

    # Count examples
    with open(data_path, "r") as f:
        num_examples = sum(1 for _ in f)
    print(f"Training data: {num_examples} examples")

    # Check if adapter exists
    adapter_path = Path("/out/qwen25_7b_tone_lora")
    if not adapter_path.exists():
        raise RuntimeError(
            f"Adapter not found at {adapter_path}. Run initial training first."
        )

    print(f"Found existing adapter at {adapter_path}")
    print(f"Files: {list(adapter_path.iterdir())}")

    # Set environment variables
    os.environ["BASE_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
    os.environ["ADAPTER_PATH"] = str(adapter_path)
    os.environ["DATA_PATH"] = data_path
    os.environ["OUTPUT_DIR"] = str(adapter_path)  # Save back to same location
    os.environ["MAX_SEQ_LEN"] = "1024"
    os.environ["BATCH_SIZE"] = "1"
    os.environ["GRAD_ACCUM"] = "16"
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["MAX_STEPS"] = str(max_steps)
    os.environ["WARMUP_RATIO"] = "0.05"
    os.environ["LOGGING_STEPS"] = "5"
    os.environ["SAVE_STEPS"] = "50"
    os.environ["NUM_EXAMPLES"] = str(num_examples)

    train_script = '''
"""Continue QLoRA training from existing adapter with W&B."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config() -> dict:
    return {
        "base_model": os.environ.get("BASE_MODEL"),
        "adapter_path": os.environ.get("ADAPTER_PATH"),
        "data_path": os.environ.get("DATA_PATH"),
        "output_dir": os.environ.get("OUTPUT_DIR"),
        "max_seq_len": int(os.environ.get("MAX_SEQ_LEN", "1024")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
        "grad_accum": int(os.environ.get("GRAD_ACCUM", "16")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "1e-4")),
        "max_steps": int(os.environ.get("MAX_STEPS", "200")),
        "warmup_ratio": float(os.environ.get("WARMUP_RATIO", "0.05")),
        "logging_steps": int(os.environ.get("LOGGING_STEPS", "5")),
        "save_steps": int(os.environ.get("SAVE_STEPS", "50")),
        "num_examples": int(os.environ.get("NUM_EXAMPLES", "0")),
    }


def load_dataset_jsonl(data_path: str, tokenizer) -> Dataset:
    """Load and format dataset for SFT."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = tokenizer.apply_chat_template(
                    data["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                examples.append({"text": text})
    return Dataset.from_list(examples)


def main():
    print("=" * 60)
    print("CONTINUED QLORA TRAINING WITH W&B")
    print("=" * 60)
    
    config = load_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize W&B
    print("\\nInitializing Weights & Biases...")
    wandb.init(
        project="idarati-tone-finetuning",
        name=f"qwen25-7b-continued-lr{config['learning_rate']}-steps{config['max_steps']}",
        config={
            "model": config["base_model"],
            "continued_from": config["adapter_path"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "grad_accum": config["grad_accum"],
            "effective_batch_size": config["batch_size"] * config["grad_accum"],
            "max_steps": config["max_steps"],
            "num_examples": config["num_examples"],
            "training_type": "continued",
        },
        tags=["qlora", "qwen2.5", "continued-training", "wandb-enabled"],
    )
    print(f"W&B run: {wandb.run.url}")
    
    print(f"\\nGPU: {torch.cuda.get_device_name(0)}")
    wandb.config.update({"gpu": torch.cuda.get_device_name(0)})
    
    print(f"\\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"\\nLoading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print(f"\\nLoading existing adapter from {config['adapter_path']}...")
    model = PeftModel.from_pretrained(
        model,
        config["adapter_path"],
        is_trainable=True,
    )
    model.print_trainable_parameters()
    
    print(f"\\nLoading dataset...")
    dataset = load_dataset_jsonl(config["data_path"], tokenizer)
    print(f"Dataset: {len(dataset)} examples")
    
    print("\\nConfiguring training...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name=f"qwen25-continued-{config['max_steps']}steps",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_len"],
        args=training_args,
        packing=False,
    )
    
    print("\\n" + "=" * 60)
    print("STARTING CONTINUED TRAINING")
    print("=" * 60)
    
    trainer.train()
    
    print("\\n" + "=" * 60)
    print("SAVING UPDATED ADAPTER")
    print("=" * 60)
    
    output_path = Path(config["output_dir"])
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Update training config
    config_path = output_path / "training_config.json"
    existing_config = {}
    if config_path.exists():
        with open(config_path) as f:
            existing_config = json.load(f)
    existing_config["continued_training"] = {
        "learning_rate": config["learning_rate"],
        "max_steps": config["max_steps"],
        "num_examples": config["num_examples"],
    }
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"\\nAdapter updated at: {output_path}")
    
    wandb.save(str(output_path / "adapter_config.json"))
    wandb.finish()
    print("W&B run finished.")
    
    print("\\n" + "=" * 60)
    print("CONTINUED TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''

    # Write and run training script
    script_path = "/tmp/continue_train.py"
    with open(script_path, "w") as f:
        f.write(train_script)

    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    volume.commit()
    print("\nVolume committed - adapter updated")

    return {"status": "success", "output_dir": "/out/qwen25_7b_tone_lora"}


@app.local_entrypoint()
def main(max_steps: int = 200, learning_rate: float = 1e-4, data_file: str = ""):
    """Continue training with W&B logging.

    Args:
        max_steps: Number of additional training steps (default: 200)
        learning_rate: Learning rate, lower for continued training (default: 1e-4)
        data_file: Path to training data JSONL file
    """
    # Find training data
    if data_file:
        data_path = Path(data_file)
    else:
        possible_paths = [
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "training"
            / "tone_sft.jsonl",
            Path("data/training/tone_sft.jsonl"),
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            raise FileNotFoundError("Training data not found")

    print(f"Loading training data from: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data_content = f.read()

    num_examples = sum(1 for line in data_content.strip().split("\n") if line)
    print(f"Loaded {num_examples} training examples")

    print(f"\nContinued training settings:")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  W&B: enabled")
    print("-" * 40)

    result = continue_training.remote(
        data_content, max_steps=max_steps, learning_rate=learning_rate
    )

    print("\n" + "=" * 60)
    print("CONTINUED TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Result: {result}")
    print("\nCheck W&B dashboard for training curves!")
