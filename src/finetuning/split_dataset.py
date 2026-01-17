"""Split generated training data into train/val/test sets."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class DatasetSplitter:
    """Split training data into train/validation/test sets."""

    def __init__(self, input_file: str):
        self.examples = self._load_jsonl(input_file)

    def _load_jsonl(self, file: str) -> List[Dict]:
        examples = []
        with open(file, "r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json.loads(line))
        return examples

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

        random.seed(seed)

        by_procedure = defaultdict(list)
        for ex in self.examples:
            context = ex["messages"][1]["content"]
            proc_id_match = re.search(r"(?:id|معرّف الإجراء):\s*([a-f0-9\-]+)", context)
            if proc_id_match:
                proc_id = proc_id_match.group(1)
                by_procedure[proc_id].append(ex)

        procedures = list(by_procedure.keys())
        random.shuffle(procedures)

        n_train = int(len(procedures) * train_ratio)
        n_val = int(len(procedures) * val_ratio)

        train_procs = procedures[:n_train]
        val_procs = procedures[n_train : n_train + n_val]
        test_procs = procedures[n_train + n_val :]

        splits = {"train": [], "val": [], "test": []}

        for proc_id in train_procs:
            splits["train"].extend(by_procedure[proc_id])

        for proc_id in val_procs:
            splits["val"].extend(by_procedure[proc_id])

        for proc_id in test_procs:
            splits["test"].extend(by_procedure[proc_id])

        return splits

    def save_splits(self, splits: Dict[str, List[Dict]], output_dir: str = "data") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_examples = sum(len(examples) for examples in splits.values())
        for split_name, examples in splits.items():
            output_file = output_path / f"{split_name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as handle:
                for ex in examples:
                    handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Saved {len(examples)} examples to {output_file}")

        info = {
            "total_examples": total_examples,
            "splits": {
                name: {
                    "count": len(examples),
                    "percentage": f"{len(examples) / total_examples * 100:.1f}%" if total_examples else "0%",
                }
                for name, examples in splits.items()
            },
        }

        info_path = output_path / "split_info.json"
        with open(info_path, "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)

        print(f"\nSplit information saved to {info_path}")
