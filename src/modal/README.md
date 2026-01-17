# QLoRA Fine-tuning for Idarati on Modal

Fine-tune Qwen2.5-7B-Instruct with QLoRA for Moroccan administrative assistant tone.

## Overview

This module trains a LoRA adapter on top of Qwen2.5-7B-Instruct to give it a:
- Friendly, professional tone
- Direct, concise responses
- Simple Arabic language style

**Important**: This fine-tuning is for **tone/style only**. Facts about procedures come from RAG (your 1700 documents).

## Quick Start

### 1. Generate Training Data

```bash
cd idarati-data-etl-for-rag

# Generate 200 tone examples using Gemini
python -m src.finetuning.modal.tone_data_generator \
  --num-examples 200 \
  --output data/training/tone_sft.jsonl
```

### 2. Run Training on Modal

```bash
cd src/finetuning/modal

# Run training (will take 30-60 minutes)
modal run modal_train.py

# Or run in background
modal run modal_train.py --detach
```

### 3. Download the Adapter

```bash
# Download adapter from Modal volume
modal volume get qwen-tone-adapter /out/qwen25_7b_tone_lora ./adapter
```

## File Structure

```
modal/
├── __init__.py
├── config.py              # Training hyperparameters
├── tone_prompts.py        # Prompt templates for data generation
├── tone_data_generator.py # Generates training data with Gemini
├── train.py               # QLoRA training script
├── modal_train.py         # Modal launcher (A100)
└── README.md              # This file
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | Qwen/Qwen2.5-7B-Instruct | Frozen |
| Quantization | 4-bit (NF4) | BitsAndBytes |
| LoRA Rank | 16 | |
| LoRA Alpha | 32 | |
| LoRA Targets | q,k,v,o,gate,up,down_proj | All attention + MLP |
| Max Seq Length | 1024 | |
| Batch Size | 1 | |
| Gradient Accumulation | 16 | Effective batch = 16 |
| Learning Rate | 2e-4 | |
| Max Steps | 500 | |
| GPU | A100 40GB | Modal |

## Data Format

Training data uses the chat format:

```jsonl
{"messages": [
  {"role": "system", "content": "أنت مساعد إداري مغربي ودود..."},
  {"role": "user", "content": "سلام، بغيت نعرف كيفاش نجدد البطاقة الوطنية"},
  {"role": "assistant", "content": "أهلاً! لتجديد البطاقة الوطنية..."}
]}
```

## Using the Adapter

### With Transformers + PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./adapter")

# Generate
tokenizer = AutoTokenizer.from_pretrained("./adapter")
```

### With vLLM (recommended for production)

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_lora=True,
    lora_modules=[{"name": "tone", "path": "./adapter"}],
)
```

## Cost Estimate

| Item | Cost |
|------|------|
| Gemini API (data gen) | Free (within limits) |
| Modal A100 (~1 hour) | ~$3 |
| **Total** | **~$3** |

## Troubleshooting

### Rate Limits
The data generator automatically rotates between 3 Gemini API keys. If you hit limits, wait a few minutes.

### Modal Errors
```bash
# Check Modal status
modal status

# View logs
modal logs qwen25-7b-tone-lora
```

### Memory Issues
If training OOMs, try:
1. Reduce `MAX_SEQ_LEN` to 512
2. Increase `GRAD_ACCUM` to 32
3. Use L4 GPU instead (more VRAM efficient)
