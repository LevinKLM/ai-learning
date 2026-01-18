"""
Run Soft Prompt Tuning training and evaluation.

This script:
1. Loads the base model with PEFT (Prompt Tuning)
2. Trains on the training set
3. Evaluates and compares with baseline
"""
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import load_tokenizer, load_and_prepare_dataset
from model import load_peft_model, print_trainable_parameters
from train import train, print_training_summary
from evaluation import evaluate, compare_results, save_results, load_results


def main():
    print("=" * 60)
    print("SOFT PROMPT TUNING")
    print("Training with PEFT Prompt Tuning")
    print("=" * 60)

    # Load configuration
    config = Config()
    print(f"\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Virtual tokens: {config.num_virtual_tokens}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Load data
    print("Loading and preparing dataset...")
    train_dataloader, eval_dataloader = load_and_prepare_dataset(config, tokenizer)
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Eval batches: {len(eval_dataloader)}")

    # Load PEFT model
    print("\nLoading model with PEFT (Prompt Tuning)...")
    model = load_peft_model(config, tokenizer)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        evaluate_fn=evaluate,
    )

    # Print training summary
    print_training_summary(history)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_result = evaluate(
        model=model,
        dataloader=eval_dataloader,
        config=config,
        model_name=f"{config.model_name} + Prompt Tuning",
        is_peft=True,
    )
    print(final_result)

    # Save results
    save_results(final_result, os.path.join(config.output_dir, "finetuned_results.json"))

    # Compare with baseline if available
    baseline_result = load_results(os.path.join(config.output_dir, "baseline_results.json"))
    if baseline_result:
        print(compare_results(baseline_result, final_result))
    else:
        print("\n⚠️  No baseline results found. Run 'python run_baseline.py' first to see comparison.")

    print("\n✅ Training complete!")
    print(f"Model saved to: {config.output_dir}/peft_model")
    print(f"Results saved to: {config.output_dir}/finetuned_results.json")

    return final_result


if __name__ == "__main__":
    main()
