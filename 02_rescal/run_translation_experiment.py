import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local utils are importable when run from project root
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
from datasets import Dataset

from misc import load_mt, load_family_splits, parse_prompt, to_float
from roh_utils import (
    fit_translational_operators,
    build_entity_vectors,
    translational_top1_accuracy,
    negation_top1_accuracy,
)


def main():
    parser = argparse.ArgumentParser(description="Run translational operator experiment for a single model")
    parser.add_argument("--repo-id", type=str, default="kdkyum/gpt_family_relation")
    parser.add_argument("--wd", type=float, required=True, help="Weight decay value (e.g., 0.0, 0.1, 1.0, ...)")
    parser.add_argument("--seed", type=int, required=True, help="Seed index (e.g., 0, 1, 2)")
    parser.add_argument("--data-repo-id", type=str, default="kdkyum/family_relation")
    parser.add_argument("--level-folder", type=str, default="lvl3_N1e+3")
    parser.add_argument("--results-dir", type=str, default="../results/translation")
    # No figures; JSON only
    args = parser.parse_args()

    subfolder = f"lvl3_lr3.0e-04_wd{args.wd}_wr0.01_s{args.seed}"

    # Prepare output dirs
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # No figure directory needed

    # Load dataset splits
    splits = load_family_splits(level_folder=args.level_folder, repo_id=args.data_repo_id)

    # Build parsed dataset from train
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

    parsed_dataset = Dataset.from_list(all_parsed_data).select(range(36*250))
    split_ds = parsed_dataset.train_test_split(test_size=18*250, train_size=18*250, shuffle=False)
    train_ds = split_ds['train']
    eval_ds = split_ds['test']

    # Build entity vectors for this single model
    mt = load_mt(repo_id=args.repo_id, subfolder=subfolder)
    v_train = build_entity_vectors(mt, train_ds, min_obs=1)
    v_test = build_entity_vectors(mt, eval_ds, min_obs=1)

    # Prepare eval pairs
    eval_pairs = []
    for i in range(len(eval_ds)):
        row = eval_ds[i]
        s, r = parse_prompt(row['prompt'])
        o = row.get('answer', [''])[0]
        eval_pairs.append((s, r, f" {o}"))

    # Fit and evaluate translational operators per layer
    results = []
    for H_LAYER in np.arange(len(v_train)):
        bs, loss = fit_translational_operators(train_ds, v_train[H_LAYER])
        acc = translational_top1_accuracy(bs, v_test[H_LAYER], eval_pairs, k_candidates=0)
        results.append((int(H_LAYER), float(loss), acc))

    # Save JSON
    translational_data = []
    for layer, loss, acc in results:
        if isinstance(acc, dict):
            for rel, acc_val in acc.items():
                translational_data.append({
                    'wd': args.wd, 'seed': args.seed, 'layer': layer, 'loss': loss,
                    'relation': rel, 'accuracy': acc_val
                })
            translational_data.append({
                'wd': args.wd, 'seed': args.seed, 'layer': layer, 'loss': loss,
                'relation': 'overall', 'accuracy': to_float(acc)
            })
        else:
            translational_data.append({
                'wd': args.wd, 'seed': args.seed, 'layer': layer, 'loss': loss,
                'relation': 'overall', 'accuracy': to_float(acc)
            })

    out_json = results_dir / f"roh_translational_accuracy_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(translational_data).to_json(out_json, orient='records', indent=4)
    print(f"Saved translational accuracy to {out_json}")

    # No plotting

    # Negation evaluation per layer
    transpose_results = []
    for H_LAYER in np.arange(len(v_test)):
        bs, _ = fit_translational_operators(train_ds, v_train[H_LAYER])
        transpose_acc = negation_top1_accuracy(bs, v_test[H_LAYER], eval_pairs, k_candidates=0)
        transpose_results.append((int(H_LAYER), transpose_acc))

    # Save negation JSON
    neg_rows = []
    for layer, acc in transpose_results:
        if isinstance(acc, dict):
            for rel, acc_val in acc.items():
                neg_rows.append({'wd': args.wd, 'seed': args.seed, 'layer': layer, 'relation': rel, 'negation_accuracy': float(acc_val)})
    out_json2 = results_dir / f"roh_translational_negation_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(neg_rows).to_json(out_json2, orient='records', indent=4)
    print(f"Saved negation results to {out_json2}")

    # No plotting


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
