# PEFT Learning Lab

Parameter-Efficient Fine-Tuning 學習實驗。

## 實驗結果

| 方法 | Accuracy | 可訓練參數 | 目錄 |
|------|----------|-----------|------|
| Baseline | 43.2% | 0 | - |
| Soft Prompt Tuning | 73.4% | 8,960 (0.002%) | `soft_prompt_tuning/` |
| LoRA (PEFT 庫) | 94.0% | 4.4M (0.88%) | `lora/` |
| LoRA (手寫) | 79.4% | 540K | `lora_from_scratch/` |

## 目錄結構

```
PEFT/
├── soft_prompt_tuning/    # Soft Prompt Tuning 實現
├── lora/                  # LoRA (使用 PEFT 庫)
└── lora_from_scratch/     # LoRA 手寫實現 (無任何庫)
```

## 運行方式

```bash
# Soft Prompt Tuning
cd soft_prompt_tuning
./venv/bin/python run_training.py

# LoRA (PEFT 庫)
cd lora
./venv/bin/python run_training.py

# LoRA (手寫)
cd lora_from_scratch
./venv/bin/python train_manual.py
```

## 學到的關鍵概念

- **Soft Prompt**: 學習可訓練的虛擬 tokens 作為輸入前綴
- **LoRA**: 用低秩矩陣分解 W + A×B 來近似權重更新
- **凍結 vs 訓練**: PEFT 方法凍結大部分參數，只訓練少量

## 模型 & 數據

- 模型: `Qwen/Qwen2.5-0.5B`
- 數據: `yelp_polarity` (情感分類)
