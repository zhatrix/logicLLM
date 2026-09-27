"""把 mlx_lm 训练的 LoRA adapter 转成 HuggingFace PEFT 格式（供 transformers / vLLM / ModelScope 使用）。

数学对应：
  MLX  : y = W x + scale * (x @ lora_a) @ lora_b      lora_a: (in, r)  lora_b: (r, out)
  PEFT : y = W x + (alpha / r) * B @ A @ x             A: (r, in)       B: (out, r)
  ⇒ A = lora_a.T, B = lora_b.T, alpha = scale * r

用法：uv run python scripts/convert_adapter_to_peft.py [--src adapters/logistics-lora] [--dst adapters/logistics-q3-14b-peft]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL_HF = "Qwen/Qwen3-14B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="adapters/logistics-lora")
    ap.add_argument("--dst", default="adapters/logistics-q3-14b-peft")
    ap.add_argument("--base", default=BASE_MODEL_HF)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    cfg = json.load(open(src / "adapter_config.json"))
    lp = cfg["lora_parameters"]
    r, scale = int(lp["rank"]), float(lp["scale"])
    alpha = scale * r

    tensors: dict[str, torch.Tensor] = {}
    modules: set[str] = set()
    with safe_open(src / "adapters.safetensors", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            # k 形如 model.layers.12.self_attn.q_proj.lora_a
            prefix, kind = k.rsplit(".", 1)
            modules.add(prefix.split(".")[-1])
            if kind == "lora_a":
                tensors[f"base_model.model.{prefix}.lora_A.weight"] = t.T.contiguous().to(torch.float32)
            elif kind == "lora_b":
                tensors[f"base_model.model.{prefix}.lora_B.weight"] = t.T.contiguous().to(torch.float32)
            else:
                raise ValueError(f"未知张量 {k}")

    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dst / "adapter_model.safetensors", metadata={"format": "pt"})
    peft_cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": a.base,
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": float(lp.get("dropout", 0.0)),
        "target_modules": sorted(modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
    }
    json.dump(peft_cfg, open(dst / "adapter_config.json", "w"), indent=2)
    n_layers = len({k.split(".")[4] for k in tensors})
    print(f"已转换 {len(tensors)} 个张量，{n_layers} 层，模块 {sorted(modules)}，r={r}，alpha={alpha} → {dst}")


if __name__ == "__main__":
    main()
