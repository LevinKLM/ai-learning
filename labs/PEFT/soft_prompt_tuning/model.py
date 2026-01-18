"""
Model setup for base model and PEFT-enabled model.
"""
import torch
from transformers import AutoModelForSequenceClassification
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model

from config import Config


def get_device(config: Config):
    """Determine the device to use."""
    if config.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.device


def load_base_model(config: Config, tokenizer):
    """
    Load the base model for sequence classification.

    Args:
        config: Configuration object
        tokenizer: The tokenizer (needed for pad_token_id)

    Returns:
        model: The base model configured for classification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,  # Binary classification (positive/negative)
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = get_device(config)
    model = model.to(device)

    return model


def load_peft_model(config: Config, tokenizer):
    """
    Load the base model and apply PEFT (Prompt Tuning).

    Args:
        config: Configuration object
        tokenizer: The tokenizer

    Returns:
        peft_model: The PEFT-enabled model
    """
    # First load the base model
    base_model = load_base_model(config, tokenizer)

    # Configure Prompt Tuning
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=config.num_virtual_tokens,
        prompt_tuning_init=PromptTuningInit.TEXT if config.prompt_tuning_init == "TEXT" else PromptTuningInit.RANDOM,
        prompt_tuning_init_text=config.prompt_tuning_init_text,
        tokenizer_name_or_path=config.model_name,
    )

    # Create PEFT model
    peft_model = get_peft_model(base_model, peft_config)

    return peft_model


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\n{'='*50}")
    print("Model Parameter Statistics")
    print(f"{'='*50}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.4f}%")
    print(f"{'='*50}\n")

    return trainable_params, all_params


if __name__ == "__main__":
    from data_loader import load_tokenizer

    config = Config()
    tokenizer = load_tokenizer(config)

    print("Loading base model...")
    base_model = load_base_model(config, tokenizer)
    print_trainable_parameters(base_model)

    print("Loading PEFT model...")
    peft_model = load_peft_model(config, tokenizer)
    print_trainable_parameters(peft_model)
