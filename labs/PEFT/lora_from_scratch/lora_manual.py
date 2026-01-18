"""
LoRA implemented from scratch - no PEFT library!

This file shows exactly what happens inside LoRA.
"""
import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    A single LoRA layer that wraps around an existing Linear layer.
    
    原理:
        output = original_linear(x) + lora_B(lora_A(x)) * scaling
        
    其中:
        lora_A: 降維 (in_features → r)
        lora_B: 升維 (r → out_features)
        scaling: alpha / r
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,  # 原始的 Linear 層
        r: int = 8,                 # LoRA 的 rank
        alpha: int = 16,            # 縮放因子
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 縮放因子
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 創建 A 和 B 矩陣
        # A: 降維 (in_features → r)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        
        # B: 升維 (r → out_features)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        # A 用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # B 初始化為 0，這樣一開始 LoRA 輸出為 0，不影響原始模型
        nn.init.zeros_(self.lora_B.weight)
        
        # 凍結原始層！這是關鍵
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        print(f"  Created LoRA layer: {in_features} → {r} → {out_features}")
        print(f"    Original params: {in_features * out_features:,} (frozen)")
        print(f"    LoRA params: {in_features * r + r * out_features:,} (trainable)")
    
    def forward(self, x):
        """
        前向傳播:
        1. 計算原始層輸出 (凍結的)
        2. 計算 LoRA 輸出 (可訓練的)
        3. 相加
        """
        # 原始輸出
        original_output = self.original_layer(x)
        
        # LoRA 輸出: x → A → dropout → B → scaling
        lora_output = self.lora_A(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_B(lora_output)
        lora_output = lora_output * self.scaling
        
        # 合併
        return original_output + lora_output


def apply_lora_to_model(model, target_modules, r=8, alpha=16, dropout=0.1):
    """
    將 LoRA 應用到模型的指定層。
    
    這就是 PEFT 庫的 get_peft_model 做的事情！
    """
    print("\n" + "="*60)
    print("Applying LoRA to model")
    print("="*60)
    
    replaced_count = 0
    
    # 遍歷所有模塊
    for name, module in model.named_modules():
        # 檢查是否是目標層
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # 找到父模塊
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                parent = model.get_submodule(parent_name) if parent_name else model
                
                # 創建 LoRA 層
                print(f"\nReplacing: {name}")
                lora_layer = LoRALayer(
                    original_layer=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                
                # 替換！
                setattr(parent, child_name, lora_layer)
                replaced_count += 1
    
    print(f"\n✅ Replaced {replaced_count} layers with LoRA")
    return model


def count_parameters(model):
    """統計可訓練和總參數數量。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*50}")
    print("Parameter Statistics")
    print(f"{'='*50}")
    print(f"Trainable: {trainable:,}")
    print(f"Total: {total:,}")
    print(f"Trainable %: {100 * trainable / total:.4f}%")
    print(f"{'='*50}")
    
    return trainable, total


# Demo
if __name__ == "__main__":
    print("Demo: LoRA from scratch")
    print("="*60)
    
    # 創建一個簡單的模型來演示
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(896, 896)
            self.layer2 = nn.Linear(896, 896)
            self.output = nn.Linear(896, 2)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.output(x)
    
    model = SimpleModel()
    print("\nBefore LoRA:")
    count_parameters(model)
    
    # 應用 LoRA
    model = apply_lora_to_model(
        model,
        target_modules=["layer1", "layer2"],
        r=8,
        alpha=16,
    )
    
    print("\nAfter LoRA:")
    count_parameters(model)
    
    # 測試前向傳播
    x = torch.randn(2, 896)
    output = model(x)
    print(f"\nTest forward pass: input {x.shape} → output {output.shape}")
