"""
Complete training script using our hand-written LoRA.
No PEFT library - everything is transparent!
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from lora_manual import apply_lora_to_model, count_parameters


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "v_proj"]  # 簡化版，只改 Q 和 V
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 500
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 256


def main():
    print("="*60)
    print("LoRA from Scratch - Training on Real Model")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # ============================================================
    # 1. 加載 Tokenizer
    # ============================================================
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ============================================================
    # 2. 加載數據
    # ============================================================
    print("\n[2/5] Loading dataset...")
    dataset = load_dataset("yelp_polarity")
    
    train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    eval_data = dataset["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))
    
    def tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = examples["label"]
        return tokens
    
    train_data = train_data.map(tokenize, batched=True, remove_columns=["text", "label"])
    eval_data = eval_data.map(tokenize, batched=True, remove_columns=["text", "label"])
    
    train_data.set_format("torch")
    eval_data.set_format("torch")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE)
    
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")
    
    # ============================================================
    # 3. 加載模型並手動應用 LoRA
    # ============================================================
    print("\n[3/5] Loading model and applying LoRA manually...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # 🔥 這是我們手寫的 LoRA！
    model = apply_lora_to_model(
        model,
        target_modules=TARGET_MODULES,
        r=LORA_R,
        alpha=LORA_ALPHA,
    )
    
    model = model.to(device)
    count_parameters(model)
    
    # ============================================================
    # 4. 訓練
    # ============================================================
    print("\n[4/5] Training...")
    
    # 只優化可訓練參數
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
        
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")
        
        # Evaluation
        model.eval()
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)
        
        print(f"Eval Acc: {eval_correct/eval_total:.4f}")
    
    # ============================================================
    # 5. 結果
    # ============================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Eval Accuracy: {eval_correct/eval_total:.4f}")
    print("\nThis was trained with OUR hand-written LoRA, not PEFT library!")


if __name__ == "__main__":
    main()
