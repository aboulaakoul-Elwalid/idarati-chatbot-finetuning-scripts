# Fine-Tuning Data Workflow

## Generate synthetic training data (Gemini)
```bash
python -m src.finetuning.cli generate \
  data/processed/ready_for_embedding.json \
  data/processed/training/training_data.jsonl \
  --api-key YOUR_GEMINI_KEY \
  --model gemini-2.5-flash
```

## Validate generated data
```bash
python -m src.finetuning.cli validate \
  data/processed/training/training_data.jsonl \
  --report data/processed/training/quality_report.json
```

## Split into train/val/test
```bash
python -m src.finetuning.cli split \
  data/processed/training/training_data.jsonl \
  --output-dir data/processed/training
```

## Run all steps in sequence
```bash
python -m src.finetuning.cli all \
  data/processed/ready_for_embedding.json \
  --api-key YOUR_GEMINI_KEY \
  --model gemini-2.5-flash
```
