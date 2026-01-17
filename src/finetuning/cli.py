"""CLI entry point for fine-tuning data generation and QA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from finetuning.config import FineTuningConfig
from finetuning.quality_control import TrainingDataValidator
from finetuning.split_dataset import DatasetSplitter
from finetuning.synthetic_data_generator import (
    DEFAULT_OPENROUTER_MODEL,
    LLAMA_SERVER_ALIASES,
    SyntheticDataGenerator,
)

MODEL_OPTIONS = {
    "1": {"label": "Gemini 2.5 Flash (Gemini API)", "model_name": "gemini-2.5-flash"},
    "2": {
        "label": f"OpenRouter: {DEFAULT_OPENROUTER_MODEL}",
        "model_name": DEFAULT_OPENROUTER_MODEL,
    },
    "3": {"label": "Local Llama server", "model_name": "llama-server"},
}

# Optional: list multiple Gemini API keys for automatic rotation on rate limits.
# Provide via environment variables (comma-separated) or `.env`.
GEMINI_API_KEYS = [
    key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(",") if key.strip()
]


def prompt_for_model(default_model: str) -> str:
    print("Choose generation model:")
    for key, option in MODEL_OPTIONS.items():
        print(f"{key}) {option['label']}")
    default_choice = next(
        (
            key
            for key, option in MODEL_OPTIONS.items()
            if option["model_name"] == default_model
        ),
        "1",
    )
    choice = (
        input(f"Enter 1-{len(MODEL_OPTIONS)} [{default_choice}]: ").strip()
        or default_choice
    )
    if choice not in MODEL_OPTIONS:
        print("Unknown choice, defaulting to the configured model.")
        choice = default_choice
    return MODEL_OPTIONS[choice]["model_name"]


def uses_openrouter(model_name: str) -> bool:
    return "/" in model_name and model_name not in LLAMA_SERVER_ALIASES


def resolve_api_key(model_name: str) -> str | None:
    if model_name in LLAMA_SERVER_ALIASES:
        return None
    if uses_openrouter(model_name):
        return os.getenv("OPENROUTER_API_KEY")
    return os.getenv("GEMINI_API_KEY")


def resolve_gemini_api_keys() -> list[str]:
    keys = [key for key in GEMINI_API_KEYS if key]
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        keys.append(env_key)
    # Preserve order while removing duplicates.
    seen = set()
    deduped = []
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning data workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate training data")
    generate_parser.add_argument("input", nargs="?", help="Procedures JSON file")
    generate_parser.add_argument("output", nargs="?", help="Output JSONL file")
    generate_parser.add_argument("--api-key", dest="api_key", help="Gemini API key")
    generate_parser.add_argument("--model", dest="model_name", help="Gemini model name")

    validate_parser = subparsers.add_parser("validate", help="Validate training data")
    validate_parser.add_argument("data", help="Training data JSONL file")
    validate_parser.add_argument(
        "--report", default="quality_report.json", help="Output report file"
    )

    split_parser = subparsers.add_parser("split", help="Split training data")
    split_parser.add_argument("data", help="Training data JSONL file")
    split_parser.add_argument(
        "--output-dir", default=None, help="Output directory for splits"
    )
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    all_parser = subparsers.add_parser("all", help="Generate + validate + split")
    all_parser.add_argument("input", nargs="?", help="Procedures JSON file")
    all_parser.add_argument("--api-key", dest="api_key", help="Gemini API key")
    all_parser.add_argument("--model", dest="model_name", help="Gemini model name")

    return parser.parse_args()


def default_paths(config: FineTuningConfig) -> tuple[Path, Path, Path]:
    input_path = Path(config.input_file)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_data.jsonl"
    metadata_path = output_dir / "training_metadata.json"
    return input_path, training_path, metadata_path


def run_generate(args: argparse.Namespace, config: FineTuningConfig) -> Path:
    input_path, training_path, metadata_path = default_paths(config)
    if args.input:
        input_path = Path(args.input)
    if args.output:
        training_path = Path(args.output)
        metadata_path = training_path.with_name(f"{training_path.stem}_metadata.json")

    generator = SyntheticDataGenerator(
        api_key=args.api_key,
        api_keys=getattr(args, "api_keys", None),
        model_name=args.model_name,
        config=config,
    )

    print("Loading procedures...")
    procedures = generator.load_procedures(input_path)
    print(f"Loaded {len(procedures)} procedures")

    print("\nGenerating training examples...")
    examples = generator.process_batch(procedures)

    print("\nSaving training data...")
    generator.save_training_data(training_path)
    generator.save_metadata(metadata_path)

    print(f"\nDone! Generated {len(examples)} examples")
    print(f"Output: {training_path}")
    return training_path


def run_validate(data_path: Path, report_path: Path) -> None:
    validator = TrainingDataValidator(str(data_path))
    validator.print_summary()
    validator.generate_report(str(report_path))


def run_split(data_path: Path, output_dir: Path, seed: int) -> None:
    splitter = DatasetSplitter(str(data_path))
    splits = splitter.split(seed=seed)
    splitter.save_splits(splits, output_dir=str(output_dir))


def main() -> None:
    args = parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    config = FineTuningConfig()

    if args.command == "generate":
        if not args.model_name:
            args.model_name = config.model_name
        api_keys = None
        if uses_openrouter(args.model_name) and not args.api_key:
            args.api_key = resolve_api_key(args.model_name)
        elif args.model_name not in LLAMA_SERVER_ALIASES:
            api_keys = [args.api_key] if args.api_key else resolve_gemini_api_keys()
        args.api_keys = api_keys
        run_generate(args, config)
        return

    if args.command == "validate":
        run_validate(Path(args.data), Path(args.report))
        return

    if args.command == "split":
        output_dir = (
            Path(args.output_dir) if args.output_dir else Path(config.output_dir)
        )
        run_split(Path(args.data), output_dir, args.seed)
        return

    if args.command == "all":
        input_path, training_path, metadata_path = default_paths(config)
        if args.input:
            input_path = Path(args.input)
        if not args.model_name:
            args.model_name = prompt_for_model(config.model_name)
        api_keys = None
        if uses_openrouter(args.model_name) and not args.api_key:
            args.api_key = resolve_api_key(args.model_name)
        elif args.model_name not in LLAMA_SERVER_ALIASES:
            api_keys = [args.api_key] if args.api_key else resolve_gemini_api_keys()
        args.api_keys = api_keys
        generator = SyntheticDataGenerator(
            api_key=args.api_key,
            api_keys=api_keys,
            model_name=args.model_name,
            config=config,
        )

        print("\n" + "=" * 70)
        print("STEP 1: GENERATING TRAINING DATA")
        print("=" * 70)
        procedures = generator.load_procedures(input_path)
        generator.process_batch(procedures)
        generator.save_training_data(training_path)
        generator.save_metadata(metadata_path)

        print("\n" + "=" * 70)
        print("STEP 2: VALIDATING QUALITY")
        print("=" * 70)
        run_validate(training_path, training_path.with_name("quality_report.json"))

        print("\n" + "=" * 70)
        print("STEP 3: SPLITTING DATASET")
        print("=" * 70)
        run_split(training_path, Path(config.output_dir), seed=42)

        print("\n" + "=" * 70)
        print("ALL OPERATIONS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print(f"  {training_path}       - Full dataset")
        print(f"  {Path(config.output_dir) / 'train.jsonl'}          - Training set")
        print(f"  {Path(config.output_dir) / 'val.jsonl'}            - Validation set")
        print(f"  {Path(config.output_dir) / 'test.jsonl'}           - Test set")
        print(
            f"  {training_path.with_name('quality_report.json')}       - Quality analysis"
        )
        print(
            f"  {Path(config.output_dir) / 'split_info.json'}      - Split statistics"
        )


if __name__ == "__main__":
    main()
