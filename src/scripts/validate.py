#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM

from olmo_core.nn.transformer import Transformer, TransformerConfig


def iter_batches_memmap(paths: List[str], chunk_size: int, batch_size: int):
    """Iterate batches of fixed-size chunks from memmap files."""
    batch = []
    for p in paths:
        arr = np.memmap(p, mode="r", dtype=np.uint32)
        n_full = (arr.shape[0] // chunk_size) * chunk_size
        for start in range(0, n_full, chunk_size):
            chunk = np.asarray(arr[start : start + chunk_size])
            batch.append(chunk)
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
    if batch:
        yield np.stack(batch, axis=0)


@torch.no_grad()
def per_instance_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    """Compute per-instance loss from logits and labels."""
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(B, T - 1)
    sum_loss_per_instance = loss.sum(dim=1)
    tokens_per_instance = torch.full((B,), T - 1, dtype=torch.long, device=loss.device)
    return sum_loss_per_instance, tokens_per_instance


def find_model_state_path(path: str) -> str:
    """Find the model state file for OLMo checkpoints."""
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(path, "model.pt"),
        os.path.join(path, "model.pth"),
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "model.bin"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not find unsharded model file under '{path}'")


def find_config_json_near(path: str) -> str:
    """Find config.json near the model checkpoint."""
    candidates = [
        os.path.join(path, "config.json"),
        os.path.join(path, "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "final", "config.json"),
        os.path.join(os.path.dirname(path.rstrip("/")), "config.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not locate 'config.json' near unsharded model at '{path}'")


@torch.no_grad()
def eval_hf(model_path: str, datasets: List[Dict[str, Any]], device: torch.device, batch_size: int, chunk_size: int):
    """Evaluate a HuggingFace model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", attn_implementation="sdpa"
    ).to(device)
    model.eval()

    totals: Dict[str, Dict[str, Any]] = {}
    overall_sum = 0.0
    overall_tok = 0

    for ds in datasets:
        lsum = 0.0
        ntok = 0
        for np_batch in iter_batches_memmap(ds["paths"], chunk_size, batch_size):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            logits = model(input_ids=ids, use_cache=False).logits
            s, n = per_instance_loss_from_logits(logits, ids)
            lsum += float(s.sum().item())
            ntok += int(n.sum().item())
        totals[ds["name"]] = {"loss": (lsum / ntok) if ntok > 0 else None, "num_tokens": ntok}
        overall_sum += lsum
        overall_tok += ntok

    return {
        "model_type": "hf",
        "overall": {"loss": (overall_sum / overall_tok) if overall_tok > 0 else None, "num_tokens": overall_tok},
        "by_label": totals,
    }


@torch.no_grad()
def eval_olmo(model_dir_or_file: str, datasets: List[Dict[str, Any]], device: torch.device, batch_size: int, chunk_size: int):
    """Evaluate an OLMo model."""
    state_path = find_model_state_path(model_dir_or_file)
    cfg_path = find_config_json_near(os.path.dirname(state_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)
    if "model" not in exp_cfg:
        raise RuntimeError(f"Invalid config at '{cfg_path}': missing 'model' section.")
    model_cfg = TransformerConfig.from_dict(exp_cfg["model"])
    model: Transformer = model_cfg.build(init_device="cpu")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device=device)
    model.eval()

    totals: Dict[str, Dict[str, Any]] = {}
    overall_sum = 0.0
    overall_tok = 0

    for ds in datasets:
        lsum = 0.0
        ntok = 0
        for np_batch in iter_batches_memmap(ds["paths"], chunk_size, batch_size):
            ids = torch.from_numpy(np_batch.astype(np.int64)).to(device)
            logits = model(input_ids=ids)
            s, n = per_instance_loss_from_logits(logits, ids)
            lsum += float(s.sum().item())
            ntok += int(n.sum().item())
        totals[ds["name"]] = {"loss": (lsum / ntok) if ntok > 0 else None, "num_tokens": ntok}
        overall_sum += lsum
        overall_tok += ntok

    return {
        "model_type": "olmo",
        "overall": {"loss": (overall_sum / overall_tok) if overall_tok > 0 else None, "num_tokens": overall_tok},
        "by_label": totals,
    }


def detect_device(device_str: Optional[str]) -> torch.device:
    """Detect device from string or auto-detect."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Simple memmap validator")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]
    chunk_size = int(cfg["chunk_size"])
    batch_size = int(cfg.get("batch_size", 8))
    device = detect_device(cfg.get("device"))
    datasets = cfg["validation_datasets"]

    if model["type"].lower() == "hf":
        result = eval_hf(model["path"], datasets, device, batch_size, chunk_size)
    elif model["type"].lower() == "olmo":
        result = eval_olmo(model["path"], datasets, device, batch_size, chunk_size)
    else:
        raise ValueError(f"Unknown model.type '{model['type']}', expected 'hf' or 'olmo'")

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
