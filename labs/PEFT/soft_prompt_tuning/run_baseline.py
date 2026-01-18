"""
Run baseline evaluation on the untrained model.

This script evaluates the base Qwen model WITHOUT any fine-tuning
to establish a baseline for comparison.
"""
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import load_tokenizer, load_and_prepare_dataset
from model import load_base_model
from evaluation import evaluate, save_results


def main():
    print("=" * 60)
    print("BASELINE EVALUATION")
    print("Evaluating the base model WITHOUT any fine-tuning")
    print("=" * 60)

    # Load configuration
    config = Config()
    print(f"\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Eval samples: {config.eval_samples}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Load data
    print("Loading and preparing dataset...")
    _, eval_dataloader = load_and_prepare_dataset(config, tokenizer)
    print(f"Eval batches: {len(eval_dataloader)}")

    # Load model
    print("\nLoading base model (this may take a moment)...")
    model = load_base_model(config, tokenizer)

    # Evaluate
    print("\nRunning evaluation...")
    result = evaluate(
        model=model,
        dataloader=eval_dataloader,
        config=config,
        model_name=f"{config.model_name} (Baseline)",
        is_peft=False,
    )

    # Print results
    print(result)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    save_results(result, os.path.join(config.output_dir, "baseline_results.json"))

    print("\n✅ Baseline evaluation complete!")
    print(f"Results saved to: {config.output_dir}/baseline_results.json")

    return result


if __name__ == "__main__":
    main()
