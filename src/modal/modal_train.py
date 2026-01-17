"""Modal launcher for QLoRA fine-tuning on A100.

Usage:
    modal run modal_train.py
    modal run modal_train.py --detach  # Run in background
"""

from pathlib import Path

import modal

# Create Modal app
app = modal.App("qwen25-7b-tone-lora")

# Create persistent volume for saving adapter
volume = modal.Volume.from_name("qwen-tone-adapter", create_if_missing=True)

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
        "wandb",  # W&B for experiment tracking
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# HuggingFace cache volume (optional, speeds up subsequent runs)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="A100",  # A100 40GB
    timeout=60 * 60 * 3,  # 3 hour timeout
    image=image,
    volumes={
        "/out": volume,
        "/cache": hf_cache,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],  # W&B API key
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # Uncomment if using private models
)
def run_training(data_content: str, use_wandb: bool = True):
    """Run the QLoRA training job on Modal.

    Args:
        data_content: The training data as a string (JSONL format)
        use_wandb: Whether to log to Weights & Biases
    """
    import os
    import subprocess

    # Write training data to file
    data_path = "/tmp/tone_sft.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(data_content)

    print(f"Written training data to {data_path}")
    print(f"Data size: {len(data_content)} bytes")

    # Count examples
    with open(data_path, "r") as f:
        num_examples = sum(1 for _ in f)
    print(f"Number of examples: {num_examples}")

    # Set environment variables for training config
    os.environ["BASE_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
    os.environ["DATA_PATH"] = data_path
    os.environ["OUTPUT_DIR"] = "/out/qwen25_7b_tone_lora"
    os.environ["MAX_SEQ_LEN"] = "1024"
    os.environ["LORA_R"] = "16"
    os.environ["LORA_ALPHA"] = "32"
    os.environ["LORA_DROPOUT"] = "0.05"
    os.environ["BATCH_SIZE"] = "1"
    os.environ["GRAD_ACCUM"] = "16"
    os.environ["LEARNING_RATE"] = "2e-4"
    os.environ["MAX_STEPS"] = "500"
    os.environ["WARMUP_RATIO"] = "0.03"
    os.environ["LOGGING_STEPS"] = "10"
    os.environ["SAVE_STEPS"] = "100"
    os.environ["USE_WANDB"] = "1" if use_wandb else "0"
    os.environ["NUM_EXAMPLES"] = str(num_examples)

    # Copy train.py to container
    train_script = '''
"""QLoRA training script for Qwen2.5-7B fine-tuning (TRL 0.8.x compatible)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# W&B import
USE_WANDB = os.environ.get("USE_WANDB", "0") == "1"
if USE_WANDB:
    import wandb


def load_config() -> dict:
    return {
        "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "data_path": os.environ.get("DATA_PATH", "/data/tone_sft.jsonl"),
        "output_dir": os.environ.get("OUTPUT_DIR", "/out/qwen25_7b_tone_lora"),
        "max_seq_len": int(os.environ.get("MAX_SEQ_LEN", "1024")),
        "lora_r": int(os.environ.get("LORA_R", "16")),
        "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
        "lora_dropout": float(os.environ.get("LORA_DROPOUT", "0.05")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
        "grad_accum": int(os.environ.get("GRAD_ACCUM", "16")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "2e-4")),
        "max_steps": int(os.environ.get("MAX_STEPS", "500")),
        "warmup_ratio": float(os.environ.get("WARMUP_RATIO", "0.03")),
        "logging_steps": int(os.environ.get("LOGGING_STEPS", "10")),
        "save_steps": int(os.environ.get("SAVE_STEPS", "100")),
        "num_examples": int(os.environ.get("NUM_EXAMPLES", "0")),
    }


def load_dataset_jsonl(data_path: str, tokenizer) -> Dataset:
    """Load and format dataset for SFT."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Format using chat template
                text = tokenizer.apply_chat_template(
                    data["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                examples.append({"text": text})
    print(f"Loaded and formatted {len(examples)} training examples")
    return Dataset.from_list(examples)


def main():
    print("=" * 60)
    print("QWEN2.5-7B QLORA FINE-TUNING")
    print("=" * 60)
    
    config = load_config()
    print("\\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize W&B
    if USE_WANDB:
        print("\\nInitializing Weights & Biases...")
        wandb.init(
            project="idarati-tone-finetuning",
            name=f"qwen25-7b-qlora-r{config['lora_r']}-lr{config['learning_rate']}",
            config={
                "model": config["base_model"],
                "lora_r": config["lora_r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": config["lora_dropout"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "grad_accum": config["grad_accum"],
                "effective_batch_size": config["batch_size"] * config["grad_accum"],
                "max_steps": config["max_steps"],
                "max_seq_len": config["max_seq_len"],
                "num_examples": config["num_examples"],
                "quantization": "4-bit NF4",
                "optimizer": "paged_adamw_8bit",
            },
            tags=["qlora", "qwen2.5", "tone-finetuning", "arabic"],
        )
        print(f"W&B run: {wandb.run.url}")
    
    if torch.cuda.is_available():
        print(f"\\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if USE_WANDB:
            wandb.config.update({"gpu": torch.cuda.get_device_name(0)})
    else:
        raise RuntimeError("CUDA not available!")
    
    print(f"\\nLoading tokenizer from {config['base_model']}...")
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
    
    print(f"\\nLoading model {config['base_model']} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    print("\\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"\\nLoading dataset from {config['data_path']}...")
    dataset = load_dataset_jsonl(config["data_path"], tokenizer)
    print(f"Dataset ready: {len(dataset)} examples")
    
    # Print first example to verify format
    print(f"\\nFirst example preview (first 200 chars):")
    print(dataset[0]["text"][:200])
    
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
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"qwen25-7b-qlora-r{config['lora_r']}" if USE_WANDB else None,
    )
    
    # SFTTrainer handles tokenization of text field
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
    print("STARTING TRAINING")
    print("=" * 60)
    
    trainer.train()
    
    print("\\n" + "=" * 60)
    print("SAVING ADAPTER")
    print("=" * 60)
    
    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    config_path = output_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\\nAdapter saved to: {output_path}")
    print(f"Files saved:")
    for file in output_path.iterdir():
        print(f"  - {file.name}")
    
    # Finish W&B run
    if USE_WANDB:
        # Log final artifacts
        wandb.save(str(output_path / "adapter_config.json"))
        wandb.finish()
        print("\\nW&B run finished and synced.")
    
    print("\\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''

    # Write training script
    script_path = "/tmp/train.py"
    with open(script_path, "w") as f:
        f.write(train_script)

    # Run training
    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Commit the volume to persist changes
    volume.commit()

    print("\n" + "=" * 60)
    print("VOLUME COMMITTED - ADAPTER SAVED")
    print("=" * 60)

    return {"status": "success", "output_dir": "/out/qwen25_7b_tone_lora"}


@app.local_entrypoint()
def main(data_file: str = "", no_wandb: bool = False):
    """Local entrypoint to start training.

    Args:
        data_file: Path to the training data JSONL file.
                   If not provided, looks for default location.
        no_wandb: Disable W&B logging (default: False, W&B enabled)
    """
    # Find training data
    if data_file:
        data_path = Path(data_file)
    else:
        # Look in default locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "training"
            / "tone_sft.jsonl",
            Path("data/training/tone_sft.jsonl"),
            Path("tone_sft.jsonl"),
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            raise FileNotFoundError(
                "Training data not found. Please provide path with --data-file "
                "or generate data first with tone_data_generator.py"
            )

    print(f"Loading training data from: {data_path}")

    # Read training data
    with open(data_path, "r", encoding="utf-8") as f:
        data_content = f.read()

    num_examples = sum(1 for line in data_content.strip().split("\n") if line)
    print(f"Loaded {num_examples} training examples")

    use_wandb = not no_wandb
    print(f"\nW&B logging: {'enabled' if use_wandb else 'disabled'}")
    print("\nStarting Modal training job...")
    print("GPU: A100 40GB")
    print("Estimated time: 30-60 minutes")
    print("-" * 40)

    # Run training on Modal
    result = run_training.remote(data_content, use_wandb=use_wandb)

    print("\n" + "=" * 60)
    print("TRAINING JOB COMPLETE!")
    print("=" * 60)
    print(f"Result: {result}")
    print("\nAdapter saved to Modal Volume: qwen-tone-adapter")
    print("Path: /out/qwen25_7b_tone_lora/")
    print("\nTo download the adapter:")
    print("  modal volume get qwen-tone-adapter /out/qwen25_7b_tone_lora ./adapter")
