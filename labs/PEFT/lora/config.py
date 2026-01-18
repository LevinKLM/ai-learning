"""
Configuration for LoRA experiment.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration for the PEFT LoRA experiment."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Dataset settings
    dataset_name: str = "yelp_polarity"
    max_length: int = 256
    train_samples: int = 1000
    eval_samples: int = 500

    # LoRA settings
    lora_r: int = 8                    # Rank of the low-rank matrices
    lora_alpha: int = 16               # Scaling factor (usually 2 * r)
    lora_dropout: float = 0.1          # Dropout for LoRA layers
    target_modules: List[str] = None   # Will be set in __post_init__

    # Training settings
    batch_size: int = 8
    learning_rate: float = 2e-4        # Lower than soft prompt, typical for LoRA
    num_epochs: int = 3
    weight_decay: float = 0.01

    # Output settings
    output_dir: str = "./outputs"
    seed: int = 42

    # Device settings
    device: str = "auto"

    def __post_init__(self):
        if self.target_modules is None:
            # Full target modules for Qwen2.5
            self.target_modules = [
                # Attention layers
                "q_proj", "k_proj", "v_proj", "o_proj",
                # FFN layers
                "gate_proj", "up_proj", "down_proj"
            ]
