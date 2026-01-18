"""
Run LoRA training and evaluation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_loader import load_tokenizer, load_and_prepare_dataset
from model import load_peft_model, print_trainable_parameters
from train import train, print_training_summary
from evaluation import evaluate, compare_results, save_results, load_results


def main():
    print("=" * 60)
    print("LoRA (Low-Rank Adaptation)")
    print("=" * 60)

    config = Config()
    print(f"\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"LoRA rank (r): {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Target modules: {config.target_modules}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    print("Loading and preparing dataset...")
    train_dataloader, eval_dataloader = load_and_prepare_dataset(config, tokenizer)
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Eval batches: {len(eval_dataloader)}")

    print("\nLoading model with LoRA...")
    model = load_peft_model(config, tokenizer)

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

    print_training_summary(history)

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_result = evaluate(
        model=model,
        dataloader=eval_dataloader,
        config=config,
        model_name=f"{config.model_name} + LoRA",
        is_peft=True,
    )
    print(final_result)

    save_results(final_result, os.path.join(config.output_dir, "lora_results.json"))

    # Compare with soft prompt tuning results if available
    soft_prompt_result = load_results("../soft_prompt_tuning/outputs/finetuned_results.json")
    if soft_prompt_result:
        print("\n" + "=" * 60)
        print("LoRA vs Soft Prompt Tuning")
        print("=" * 60)
        print(compare_results(soft_prompt_result, final_result))

    print("\n✅ Training complete!")
    print(f"Model saved to: {config.output_dir}/lora_model")

    return final_result


if __name__ == "__main__":
    main()
