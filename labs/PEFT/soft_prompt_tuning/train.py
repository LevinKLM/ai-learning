"""
Training script for Soft Prompt Tuning with best model checkpoint.
"""
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import copy

from config import Config
from model import get_device, print_trainable_parameters


def train(model, train_dataloader, eval_dataloader, config: Config, evaluate_fn=None):
    """
    Train the PEFT model using Soft Prompt Tuning.

    Args:
        model: The PEFT model to train
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        config: Configuration object
        evaluate_fn: Optional evaluation function to call after each epoch

    Returns:
        training_history: List of dicts with loss/accuracy per epoch
        best_model: The model with best eval accuracy
    """
    device = get_device(config)
    model.train()

    # Only optimize the trainable parameters (soft prompts)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Print trainable parameters
    print_trainable_parameters(model)

    training_history = []
    best_eval_acc = 0.0
    best_model_state = None

    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })

        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = correct / total

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": train_accuracy,
        }

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")

        # Evaluate if function provided
        if evaluate_fn is not None and eval_dataloader is not None:
            eval_result = evaluate_fn(model, eval_dataloader, config, is_peft=True)
            epoch_result["eval_loss"] = eval_result.avg_loss
            epoch_result["eval_accuracy"] = eval_result.accuracy
            print(f"  Eval Loss: {eval_result.avg_loss:.4f}")
            print(f"  Eval Accuracy: {eval_result.accuracy:.4f}")

            # Save best model checkpoint
            if eval_result.accuracy > best_eval_acc:
                best_eval_acc = eval_result.accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  ⭐ New best model! Accuracy: {best_eval_acc:.4f}")

        training_history.append(epoch_result)

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model with eval accuracy: {best_eval_acc:.4f}")

    # Save the best model
    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, "peft_model")
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    return training_history


def print_training_summary(history):
    """Print a summary of the training history."""
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Eval Loss':<12} {'Eval Acc':<12}")
    print("-" * 60)

    for h in history:
        eval_loss = h.get("eval_loss", "N/A")
        eval_acc = h.get("eval_accuracy", "N/A")
        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        if isinstance(eval_acc, float):
            eval_acc = f"{eval_acc:.4f}"

        print(f"{h['epoch']:<8} {h['train_loss']:<12.4f} {h['train_accuracy']:<12.4f} {eval_loss:<12} {eval_acc:<12}")

    print(f"{'='*60}")
