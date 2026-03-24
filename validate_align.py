#!/usr/bin/env python3
"""Quick validation of Stage 5 SFT model."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[验证] 加载 Stage 5 SFT 模型...")
model_path = "results/align/final_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[验证] 模型已加载到 {device}")

    # Test generation
    test_prompt = "What is machine learning?"
    print(f"[验证] 测试生成: '{test_prompt}'")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[验证] 生成结果: '{generated}'")
    print("[验证] Stage 5 Align 成功!")

except Exception as e:
    print(f"[错误] 验证失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
