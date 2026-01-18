"""
Configuration for Soft Prompt Tuning experiment.
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the PEFT soft prompt tuning experiment."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Dataset settings
    dataset_name: str = "yelp_polarity"
    max_length: int = 256
    train_samples: int = 1000  # Use subset for faster experimentation
    eval_samples: int = 500

    # Soft Prompt settings
    num_virtual_tokens: int = 8  # Number of learnable prompt tokens
    prompt_tuning_init: str = "TEXT"  # Initialize from text or random
    prompt_tuning_init_text: str = "Classify the sentiment of this review:"

    # Training settings
    batch_size: int = 8
    learning_rate: float = 1e-2  # Reduced from 3e-2 to prevent overfitting
    num_epochs: int = 3
    weight_decay: float = 0.01

    # Output settings
    output_dir: str = "./outputs"
    seed: int = 42

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
