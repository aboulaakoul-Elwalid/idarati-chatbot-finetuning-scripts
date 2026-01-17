from dataclasses import dataclass
from pathlib import Path

from src import config as root_config


@dataclass
class FineTuningConfig:
    input_file: Path = root_config.DATA_PROCESSED / "ready_for_embedding.json"
    output_dir: Path = root_config.DATA_PROCESSED / "training"
    checkpoint_dir: Path = root_config.DATA_PROCESSED / "training" / "checkpoints"
    checkpoint_every: int = 5
    resume_from_checkpoint: bool = True
    llama_server_url: str = "http://localhost:8080"
    llama_server_model: str = "local"
    system_prompt: str = (
        "أنت مساعد إداري متخصص في مساعدة المواطنين المغاربة في الإجراءات الإدارية. "
        "أجب بوضوح ودقة بناءً على المعلومات المقدمة فقط."
    )
    model_name: str = "gemini-2.5-flash"
    max_output_tokens: int = 4096
    temperature: float = 0.9
    examples_per_procedure: int = 7
    delay_between_calls: float = 0.6
