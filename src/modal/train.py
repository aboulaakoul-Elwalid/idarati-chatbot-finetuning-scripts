"""QLoRA training script for Qwen2.5-7B fine-tuning.

This script is designed to run on Modal with A100 GPU.
It trains a LoRA adapter on top of the frozen base model.
"""

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
from trl import SFTTrainer


def load_config() -> dict:
    """Load training configuration from environment variables."""
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
    }


def load_dataset(data_path: str) -> Dataset:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def format_chat(example: dict, tokenizer) -> dict:
    """Format chat messages using the tokenizer's chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    """Main training function."""
    print("=" * 60)
    print("QWEN2.5-7B QLORA FINE-TUNING")
    print("=" * 60)

    # Load configuration
    config = load_config()
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        raise RuntimeError("CUDA not available!")

    # Load tokenizer
    print(f"\nLoading tokenizer from {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model {config['base_model']} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print(f"\nLoading dataset from {config['data_path']}...")
    dataset = load_dataset(config["data_path"])

    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    print(f"Dataset ready: {len(dataset)} examples")

    # Training arguments
    print("\nConfiguring training...")
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
        report_to="none",
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_len"],
        args=training_args,
        packing=False,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    # Save adapter
    print("\n" + "=" * 60)
    print("SAVING ADAPTER")
    print("=" * 60)

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save training config
    config_path = output_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAdapter saved to: {output_path}")
    print(f"Files saved:")
    for file in output_path.iterdir():
        print(f"  - {file.name}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
