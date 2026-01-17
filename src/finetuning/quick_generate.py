"""Quick-start script for generating synthetic training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from finetuning.synthetic_data_generator import SyntheticDataGenerator


def quick_generate(
    input_json_file: str, output_jsonl_file: str, api_key: str | None = None
) -> None:
    """Generate training data in one step."""
    generator = SyntheticDataGenerator(api_key=api_key)

    print("Loading procedures...")
    procedures = generator.load_procedures(input_json_file)
    print(f"Loaded {len(procedures)} procedures")

    print("\nGenerating training examples...")
    examples = generator.process_batch(procedures)

    print("\nSaving training data...")
    output_path = Path(output_jsonl_file)
    generator.save_training_data(output_path)
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    generator.save_metadata(metadata_path)

    print(f"\nDone! Generated {len(examples)} examples")
    print(f"Output: {output_jsonl_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data quickly"
    )
    parser.add_argument("input", help="Input procedures JSON file")
    parser.add_argument("output", help="Output JSONL file for training data")
    parser.add_argument("--api-key", dest="api_key", help="Gemini API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quick_generate(args.input, args.output, args.api_key)


if __name__ == "__main__":
    main()
