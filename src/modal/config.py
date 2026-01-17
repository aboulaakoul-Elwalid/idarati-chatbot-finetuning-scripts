"""Training configuration for QLoRA fine-tuning on Modal."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning of Qwen2.5-7B."""

    # Model
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training parameters
    max_seq_len: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_steps: int = 500
    warmup_ratio: float = 0.03
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 100

    # Paths (for Modal)
    data_path: str = "/data/tone_sft.jsonl"
    output_dir: str = "/out/qwen25_7b_tone_lora"

    # System prompt for the fine-tuned model
    system_prompt: str = (
        "أنت مساعد إداري مغربي ودود. تساعد المواطنين في الإجراءات الإدارية بأسلوب بسيط ومباشر. "
        "أجب بعربية بسيطة وسهلة الفهم. كن مختصراً ومفيداً. "
        "إذا لم تعرف معلومة، قل ذلك بوضوح. لا تخترع معلومات."
    )


# Mistral model for data generation
MISTRAL_MODEL = "mistral-small-latest"

# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "training"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_DIR / "tone_sft.jsonl"
