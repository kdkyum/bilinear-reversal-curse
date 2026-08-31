"""Shared helpers: model/tokenizer loading, dataset splits, prompt parsing."""

import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "kdkyum/gpt_family_relation"
DATA_REPO = "kdkyum/family_relation"


def to_float(x):
    try:
        return float(x)
    except Exception:
        if isinstance(x, dict):
            return float(np.mean([to_float(v) for v in x.values()])) if x else float("nan")
        if isinstance(x, (list, tuple)):
            return float(np.mean([to_float(v) for v in x])) if x else float("nan")
        if isinstance(x, torch.Tensor):
            return float(x.item())
        return float("nan")


@dataclass
class ModelAndTokenizer:
    model: nn.Module
    tokenizer: any


def load_mt(repo_id: str = MODEL_REPO, subfolder: str = "", device: str = None) -> ModelAndTokenizer:
    """Load one trained model + tokenizer from the HuggingFace Hub."""
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder)
    tok = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    model.eval()
    return ModelAndTokenizer(model=model, tokenizer=tok)


def load_family_splits(level_folder: str = "lvl3_N1e+3", repo_id: str = DATA_REPO):
    """Load the family_relation dataset splits from the HuggingFace Hub.

    train rows look like {"train": "<one family document>"}; eval rows like
    {"prompt": ..., "answer": [...]}.
    """
    def fetch(fname, key):
        path = hf_hub_download(repo_id, f"{level_folder}/{fname}", repo_type="dataset")
        with open(path) as f:
            return json.load(f)[key]

    return {
        "reverse_bi": Dataset.from_list(fetch("eval_reverse_bi.json", "reverse_bi")),
        "reverse_uni": Dataset.from_list(fetch("eval_reverse_uni.json", "reverse_uni")),
        "train": Dataset.from_dict({"train": fetch("train.json", "train")}),
    }


def parse_prompt(prompt: str) -> Tuple[str, str]:
    """Split a prompt like ' Steven Harry Ramsey mother' into (' Steven Harry Ramsey', 'mother')."""
    parts = prompt.split()
    if not parts:
        raise ValueError(f"Empty prompt: {prompt!r}")
    return f" {' '.join(parts[:-1])}", parts[-1]
