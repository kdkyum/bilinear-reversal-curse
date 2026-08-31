import argparse
import os
import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from misc import load_mt, load_family_splits, to_float
from lre_utils import fit_jacobian_hz_1order, lre_top1_accuracy


def main():
    parser = argparse.ArgumentParser(description="Run LRE Jacobian experiment (ridge) for a single model")
    parser.add_argument("--repo-id", type=str, default="kdkyum/gpt_family_relation")
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-repo-id", type=str, default="kdkyum/family_relation")
    parser.add_argument("--level-folder", type=str, default="lvl3_N1e+3")
    parser.add_argument("--results-dir", type=str, default="../results/lre_jacobian")
    parser.add_argument("--train_size", type=int, default=-1, help="Training examples per relation for the Jacobian estimate; -1 for all.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    subfolder = f"lvl3_lr3.0e-04_wd{args.wd}_wr0.01_s{args.seed}"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    splits = load_family_splits(level_folder=args.level_folder, repo_id=args.data_repo_id)

    # Build dataset from train text into prompt/answer pairs similar to other scripts
    all_parsed_data = []
    seen_prompts = set()
    for example in splits["train"]:
        for item in example["train"].split(". "):
            if item.strip():
                parts = item.strip().split()
                if len(parts) >= 6:
                    prompt = " ".join(parts[:4])
                    answer = " ".join(parts[-3:])
                    if answer.endswith('.'):
                        answer = answer[:-1]
                    if prompt not in seen_prompts:
                        all_parsed_data.append({"prompt": f" {prompt}", "answer": [answer]})
                        seen_prompts.add(prompt)

    from datasets import Dataset
    parsed_dataset = Dataset.from_list(all_parsed_data).select(range(36*250))
    split_ds = parsed_dataset.train_test_split(test_size=18*250, train_size=18*250, shuffle=False)
    train_ds = split_ds['train']
    eval_ds = split_ds['test']

    mt = load_mt(repo_id=args.repo_id, subfolder=subfolder)
    logging.info(mt.model)

    ms_models = []
    rows = []
    for H_LAYER in tqdm(range(mt.model.config.num_hidden_layers)):
        Ms = fit_jacobian_hz_1order(mt, train_ds, H_LAYER, args.train_size)
        ms_models.append(Ms)
    
    betas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for H_LAYER, Ms in enumerate(ms_models):
        acc_by_beta = lre_top1_accuracy(mt, Ms, eval_ds, H_LAYER, betas=betas)
        for beta, acc_dict in acc_by_beta.items():
            for rel, acc_val in acc_dict.items():
                rows.append(
                    {
                        "wd": args.wd,
                        "seed": args.seed,
                        "layer": int(H_LAYER),
                        "relation": rel,
                        "beta": beta,
                        "accuracy": to_float(acc_val),
                    }
                )
            logging.info(f"Layer {int(H_LAYER)} completed for β={beta}")

    out_json = results_dir / f"lre_jacobian_accuracy_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(rows).to_json(out_json, orient='records', indent=4)
    logging.info(f"Saved LRE results to {out_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
