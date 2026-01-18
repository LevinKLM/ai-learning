"""
Evaluation framework for measuring model performance.

This is the key component for comparing baseline vs fine-tuned models.
"""
import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import json
from datetime import datetime

from config import Config
from model import get_device


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    avg_loss: float
    total_samples: int
    correct_predictions: int
    model_name: str
    is_peft: bool
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "avg_loss": self.avg_loss,
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "model_name": self.model_name,
            "is_peft": self.is_peft,
            "timestamp": self.timestamp,
        }

    def __str__(self):
        return f"""
{'='*50}
Evaluation Results
{'='*50}
Model: {self.model_name}
PEFT Enabled: {self.is_peft}
Accuracy: {self.accuracy:.4f} ({self.correct_predictions}/{self.total_samples})
Average Loss: {self.avg_loss:.4f}
Timestamp: {self.timestamp}
{'='*50}
"""


def evaluate(model, dataloader, config: Config, model_name: str = "", is_peft: bool = False) -> EvaluationResult:
    """
    Evaluate a model on the given dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader containing evaluation data
        config: Configuration object
        model_name: Name of the model for logging
        is_peft: Whether this is a PEFT model

    Returns:
        EvaluationResult containing accuracy, loss, and metadata
    """
    device = get_device(config)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return EvaluationResult(
        accuracy=accuracy,
        avg_loss=avg_loss,
        total_samples=total,
        correct_predictions=correct,
        model_name=model_name or config.model_name,
        is_peft=is_peft,
    )


def compare_results(baseline: EvaluationResult, finetuned: EvaluationResult) -> str:
    """
    Compare baseline and fine-tuned results.

    Args:
        baseline: Evaluation result from baseline model
        finetuned: Evaluation result from fine-tuned model

    Returns:
        Formatted comparison string
    """
    acc_diff = finetuned.accuracy - baseline.accuracy
    loss_diff = finetuned.avg_loss - baseline.avg_loss

    comparison = f"""
{'='*60}
COMPARISON: Baseline vs Fine-tuned
{'='*60}

                    Baseline        Fine-tuned      Change
----------------------------------------------------------
Accuracy:           {baseline.accuracy:.4f}          {finetuned.accuracy:.4f}          {acc_diff:+.4f}
Avg Loss:           {baseline.avg_loss:.4f}          {finetuned.avg_loss:.4f}          {loss_diff:+.4f}

Improvement:
  - Accuracy: {acc_diff * 100:+.2f}% points
  - Loss reduced by: {-loss_diff:.4f}

{'='*60}
"""
    return comparison


def save_results(result: EvaluationResult, filepath: str):
    """Save evaluation results to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Optional[EvaluationResult]:
    """Load evaluation results from a JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return EvaluationResult(**data)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    # Demo the evaluation result formatting
    demo_baseline = EvaluationResult(
        accuracy=0.5123,
        avg_loss=0.6931,
        total_samples=500,
        correct_predictions=256,
        model_name="Qwen/Qwen2.5-0.5B",
        is_peft=False,
    )

    demo_finetuned = EvaluationResult(
        accuracy=0.7890,
        avg_loss=0.4521,
        total_samples=500,
        correct_predictions=394,
        model_name="Qwen/Qwen2.5-0.5B + Prompt Tuning",
        is_peft=True,
    )

    print(demo_baseline)
    print(demo_finetuned)
    print(compare_results(demo_baseline, demo_finetuned))
