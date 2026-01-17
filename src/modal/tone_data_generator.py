"""Generate tone/style training data using Mistral AI API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from mistralai import Mistral

from .config import DEFAULT_OUTPUT_FILE, TrainingConfig
from .tone_prompts import SEED_EXAMPLES, build_generation_prompt

# Load environment variables from .env file
load_dotenv()

# JSON schema for structured output
EXAMPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"],
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "category": {"type": "string"},
                    "procedure": {"type": "string"},
                },
                "required": ["messages"],
            },
        }
    },
    "required": ["examples"],
}


class ToneDataGenerator:
    """Generate conversational tone training data using Mistral AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mistral-small-latest",
    ) -> None:
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("MISTRAL_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_KEY in .env or pass api_key parameter."
            )

        self.model_name = model_name
        self.config = TrainingConfig()
        self.generated_examples: List[Dict] = []

    def generate_batch(
        self,
        num_examples: int = 10,
        max_retries: int = 3,
    ) -> List[Dict]:
        """Generate a batch of training examples."""
        prompt = build_generation_prompt(num_examples)

        for attempt in range(max_retries):
            try:
                with Mistral(api_key=self.api_key) as client:
                    response = client.chat.complete(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.9,
                        max_tokens=8192,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "tone_examples",
                                "schema": EXAMPLES_SCHEMA,
                            },
                        },
                    )

                # Parse the structured JSON response
                content = response.choices[0].message.content
                data = json.loads(content)
                examples = data.get("examples", [])

                # Validate examples
                valid_examples = [ex for ex in examples if self._validate_example(ex)]
                return valid_examples

            except Exception as exc:
                error_str = str(exc)
                print(f"  Attempt {attempt + 1} failed: {error_str[:150]}")

                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue

                print(f"  Failed after {max_retries} attempts")
                return []

        return []

    def _validate_example(self, example: Dict) -> bool:
        """Validate that an example has the correct structure."""
        if "messages" not in example:
            return False

        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # Check for required roles
        roles = [m.get("role") for m in messages]
        if "user" not in roles or "assistant" not in roles:
            return False

        # Ensure all messages have content
        for msg in messages:
            if not msg.get("content"):
                return False

        return True

    def _append_to_file(
        self, examples: List[Dict], output_path: Path, clear: bool = False
    ) -> None:
        """Append examples to JSONL file."""
        mode = "w" if clear else "a"
        with open(output_path, mode, encoding="utf-8") as f:
            for example in examples:
                training_item = {"messages": example["messages"]}
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")

    def generate_dataset(
        self,
        total_examples: int = 200,
        batch_size: int = 10,
        delay_between_batches: float = 2.0,
        include_seeds: bool = True,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
        """Generate the full training dataset with incremental saving."""
        all_examples = []
        output_path = Path(output_path) if output_path else DEFAULT_OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear file at start
        output_path.write_text("")

        # Start with seed examples
        if include_seeds:
            all_examples.extend(SEED_EXAMPLES)
            print(f"Added {len(SEED_EXAMPLES)} seed examples")
            # Save seeds immediately
            self._append_to_file(SEED_EXAMPLES, output_path)

        remaining = total_examples - len(all_examples)
        num_batches = (remaining + batch_size - 1) // batch_size

        print(f"\nGenerating {remaining} examples in {num_batches} batches...")
        print(f"Model: {self.model_name}")
        print(f"Saving incrementally to: {output_path}")

        for i in range(num_batches):
            batch_num = i + 1
            examples_needed = min(batch_size, remaining - (i * batch_size))

            print(
                f"\nBatch {batch_num}/{num_batches}: Generating {examples_needed} examples..."
            )

            batch = self.generate_batch(num_examples=examples_needed)

            if batch:
                all_examples.extend(batch)
                # Save incrementally after each batch
                self._append_to_file(batch, output_path)
                print(f"  Got {len(batch)} examples (total: {len(all_examples)})")
            else:
                print(f"  Batch {batch_num} failed, continuing...")

            # Delay between batches to respect rate limits
            if i < num_batches - 1:
                print(f"  Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)

        self.generated_examples = all_examples
        return all_examples

    def save_dataset(self, output_path: Optional[Path] = None) -> Path:
        """Save the generated dataset to JSONL file."""
        output_path = output_path or DEFAULT_OUTPUT_FILE
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in self.generated_examples:
                # Only save the messages, not metadata
                training_item = {"messages": example["messages"]}
                f.write(json.dumps(training_item, ensure_ascii=False) + "\n")

        print(f"\nSaved {len(self.generated_examples)} examples to {output_path}")
        return output_path

    def save_with_metadata(self, output_path: Optional[Path] = None) -> Path:
        """Save dataset with metadata for analysis."""
        output_path = output_path or DEFAULT_OUTPUT_FILE.with_suffix(".full.jsonl")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in self.generated_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Saved full dataset with metadata to {output_path}")
        return output_path


def main():
    """CLI entry point for tone data generation."""
    parser = argparse.ArgumentParser(
        description="Generate tone training data using Mistral AI"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=200,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Examples per API call",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=2.0,
        help="Delay between batches (seconds)",
    )
    parser.add_argument(
        "--no-seeds",
        action="store_true",
        help="Don't include seed examples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-small-latest",
        help="Mistral model to use (default: mistral-small-latest)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TONE DATA GENERATOR (Mistral AI)")
    print("=" * 60)
    print(f"Target: {args.num_examples} examples")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 60)

    output_path = Path(args.output)
    generator = ToneDataGenerator(model_name=args.model)

    generator.generate_dataset(
        total_examples=args.num_examples,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        include_seeds=not args.no_seeds,
        output_path=output_path,
    )

    # Save metadata version
    generator.save_with_metadata(output_path.with_suffix(".full.jsonl"))

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(generator.generated_examples)}")
    print(f"Training file: {output_path}")


if __name__ == "__main__":
    main()
