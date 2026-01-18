"""
Data loading and preprocessing for Yelp Polarity dataset.
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config


def load_tokenizer(config: Config):
    """Load the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_prepare_dataset(config: Config, tokenizer):
    """
    Load Yelp Polarity dataset and prepare it for training/evaluation.

    Returns:
        train_dataloader, eval_dataloader
    """
    # Load dataset
    dataset = load_dataset(config.dataset_name)

    # Sample subsets for faster experimentation
    train_dataset = dataset["train"].shuffle(seed=config.seed).select(
        range(config.train_samples)
    )
    eval_dataset = dataset["test"].shuffle(seed=config.seed).select(
        range(config.eval_samples)
    )

    def tokenize_function(examples):
        """Tokenize the text and prepare labels."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors=None,
        )
        # Labels are already 0 (negative) or 1 (positive) in yelp_polarity
        tokenized["labels"] = examples["label"]
        return tokenized

    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
    )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    # Test the data loading
    config = Config()
    tokenizer = load_tokenizer(config)
    train_loader, eval_loader = load_and_prepare_dataset(config, tokenizer)

    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")

    # Show a sample
    batch = next(iter(train_loader))
    print(f"\nSample batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels: {batch['labels']}")
