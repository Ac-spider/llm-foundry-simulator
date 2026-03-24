#!/usr/bin/env python3
"""
对比不同学习率训练模型的生成效果
使用三个不同的checkpoint：
- lr=1e-3: results/train_2k_lr1e3/66cf7fd3/checkpoints/step_002000.pt
- lr=5e-4: results/train_2k_lr5e4/e4c8ee0a/checkpoints/step_002000.pt
- lr=1e-4: results/train_2k_lr1e4/4107b595/checkpoints/step_002000.pt
"""

import sys
from pathlib import Path
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_foundry.common.model import BasicsTransformerLM, ModelConfig, create_model
from llm_foundry.stage1_tokenize import BPETokenizer

# 模型配置
model_config = ModelConfig(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    rope_theta=10000.0,
)

# 三个不同学习率的checkpoint路径
checkpoints = {
    "lr=1e-3 (best)": "results/train_2k_lr1e3/66cf7fd3/checkpoints/step_002000.pt",
    "lr=5e-4 (mid)": "results/train_2k_lr5e4/e4c8ee0a/checkpoints/step_002000.pt",
    "lr=1e-4 (worst)": "results/train_2k_lr1e4/4107b595/checkpoints/step_002000.pt",
}

# 测试提示
test_prompts = [
    "The most important thing in life is",
    "In the future, AI will",
    "Once upon a time",
    "The Earth is a planet that",
]

def load_model(checkpoint_path: str):
    """加载模型和权重"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_config, use_flash_attn=False)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, device

def generate_text(model, tokenizer, device, prompt: str, max_tokens: int = 50, temperature: float = 1.0):
    """生成文本"""
    # 编码prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_tokens):
            # 截取最后context_length个token
            if len(generated) > model_config.context_length:
                input_tensor = torch.tensor([generated[-model_config.context_length:]], dtype=torch.long, device=device)
            else:
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)

            # 前向传播
            logits = model(input_tensor)

            # 取最后一个token的logits
            next_token_logits = logits[0, -1, :] / temperature

            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            # 可以在这里添加停止条件，比如遇到特定token或达到最大长度
            # 暂时不检查EOS，因为我们不确定eos_token_id

    return tokenizer.decode(generated)

def main():
    # 加载tokenizer
    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer_path = "results/tokenizer/bpe_tokenizer"
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    print("=" * 80)

    # 加载三个模型
    models = {}
    for name, path in checkpoints.items():
        print(f"\nLoading model: {name}")
        print(f"  Checkpoint: {path}")
        try:
            model, device = load_model(path)
            models[name] = (model, device)
            print(f"  ✓ Loaded successfully")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)

    # 对每个prompt生成
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*80}")

        for name, (model, device) in models.items():
            print(f"\n  [{name}]")
            try:
                result = generate_text(model, tokenizer, device, prompt, max_tokens=40, temperature=0.8)
                print(f"  → {result}")
            except Exception as e:
                print(f"  → Error: {e}")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
