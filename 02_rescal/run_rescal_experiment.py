import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
from datasets import Dataset

from misc import load_mt, load_family_splits, parse_prompt, to_float
from roh_utils import (
    fit_rescal_operators,
    bilinear_top1_accuracy,
    build_entity_vectors,
    compositional_top1_accuracy,
    transpose_top1_accuracy,
)


def main():
    parser = argparse.ArgumentParser(description="Run RESCAL experiment (bilinear + composition + transpose) for a single model")
    parser.add_argument("--repo-id", type=str, default="kdkyum/gpt_family_relation")
    parser.add_argument("--wd", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-repo-id", type=str, default="kdkyum/family_relation")
    parser.add_argument("--level-folder", type=str, default="lvl3_N1e+3")
    parser.add_argument("--results-dir", type=str, default="../results/rescal")
    parser.add_argument("--N", type=int, default=-1, help="Number of pairs per relation to use in training; -1 for all.")

    args = parser.parse_args()

    subfolder = f"lvl3_lr3.0e-04_wd{args.wd}_wr0.01_s{args.seed}"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # No figure directory needed

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

    # Build entity vectors
    mt = load_mt(repo_id=args.repo_id, subfolder=subfolder)
    v_train = build_entity_vectors(mt, train_ds, min_obs=1)
    v_eval = build_entity_vectors(mt, eval_ds, min_obs=1)

    # Prepare eval pairs early (used in HPO loop)
    eval_pairs = []
    for i in range(len(eval_ds)):
        row = eval_ds[i]
        s, r = parse_prompt(row['prompt'])
        o = row.get('answer', [''])[0]
        eval_pairs.append((s, r, f" {o}"))

    # l2 hyperparameter search; select best (max 'overall' over layers)
    # L2_CANDIDATES = np.logspace(-2, 0, num=11, base=10)
    L2_CANDIDATES = np.logspace(-3, -1, num=21, base=10)
    best_overall = -1.0
    best = {"l2": None, "layer": None, "Ms": None}
    print(f"Searching l2 candidates: {L2_CANDIDATES}")
    Ms_cache = {}  # cache Ms per (l2, layer)
    for l2 in L2_CANDIDATES:
        Ms_cache[float(l2)] = {}
        for H_LAYER in np.arange(len(v_train)):
            Ms, _ = fit_rescal_operators(train_ds, v_train[H_LAYER], l2=l2, max_pairs_per_rel=args.N)
            Ms_cache[float(l2)][int(H_LAYER)] = Ms
            acc = bilinear_top1_accuracy(Ms, v_eval[int(H_LAYER)], eval_pairs, k_candidates=0)
            overall = float(to_float(acc))
            if overall > best_overall:
                best_overall = overall
                best = {"l2": float(l2), "layer": int(H_LAYER), "Ms": Ms}
    print(f"Best setting -> l2={best['l2']}, layer={best['layer']}, overall={best_overall:.4f}")

    # Reuse cached operators for all layers using the best l2 (no refit)
    best_l2 = best["l2"]
    best_layer = best["layer"]
    bestl2_Ms_by_layer = Ms_cache[float(best_l2)]

    # Bilinear top-1 for all layers (best l2)
    bilinear_rows = []
    for H_LAYER, Ms in bestl2_Ms_by_layer.items():
        acc = bilinear_top1_accuracy(Ms, v_eval[H_LAYER], eval_pairs, k_candidates=0)
        if isinstance(acc, dict):
            for rel, acc_val in acc.items():
                bilinear_rows.append({
                    'wd': args.wd, 'seed': args.seed, 'l2': best_l2, 'layer': H_LAYER,
                    'relation': rel, 'accuracy': float(acc_val)
                })
        bilinear_rows.append({
            'wd': args.wd, 'seed': args.seed, 'l2': best_l2, 'layer': H_LAYER,
            'relation': 'overall', 'accuracy': float(to_float(acc))
        })

    out_json = results_dir / f"rescal_accuracy_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(bilinear_rows).to_json(out_json, orient='records', indent=4)
    print(f"Saved bilinear accuracy (best l2 across all layers) to {out_json}")

    # Composition evaluation helper (unchanged)
    def test_composition(Ms, v, eval_ds, layer, composition):
        composition_eval_pairs = []
        for i in range(len(eval_ds)):
            row = eval_ds[i]
            s, r = parse_prompt(row['prompt'])
            o = row.get('answer', [''])[0]
            rel, target = composition
            if r == target:
                composition_eval_pairs.append((s, rel, f" {o}"))
        acc = compositional_top1_accuracy(Ms, v[layer], composition_eval_pairs, k_candidates=0)
        return acc

    composition_cases = {
        "mother-husband": (("mother", "husband"), "father"),
        "father-wife": (("father", "wife"), "mother"),
        "son-sister": (("son", "sister"), "daughter"),
        "daughter-brother": (("daughter", "brother"), "son"),
    }

    # Composition for all layers (best l2)
    comp_rows = []
    for H_LAYER, Ms in bestl2_Ms_by_layer.items():
        for name, (rels, target) in composition_cases.items():
            acc = test_composition(Ms, v_eval, eval_ds, H_LAYER, (rels, target))
            comp_rows.append({
                'wd': args.wd, 'seed': args.seed, 'l2': best_l2, 'layer': H_LAYER,
                'composition': name, 'accuracy': float(acc)
            })

    out_json2 = results_dir / f"rescal_composition_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(comp_rows).to_json(out_json2, orient='records', indent=4)
    print(f"Saved composition accuracy (best l2 across all layers) to {out_json2}")

    # Transpose for all layers (best l2)
    transpose_rows = []
    for H_LAYER, Ms in bestl2_Ms_by_layer.items():
        transpose_acc = transpose_top1_accuracy(Ms, v_eval[H_LAYER], eval_pairs, k_candidates=0)
        if isinstance(transpose_acc, dict):
            for rel, acc_val in transpose_acc.items():
                transpose_rows.append({
                    'wd': args.wd, 'seed': args.seed, 'l2': best_l2, 'layer': H_LAYER,
                    'relation': rel, 'transpose_accuracy': float(acc_val)
                })

    out_json3 = results_dir / f"rescal_transpose_wd{args.wd}_s{args.seed}.json"
    pd.DataFrame(transpose_rows).to_json(out_json3, orient='records', indent=4)
    print(f"Saved transpose results (best l2 across all layers) to {out_json3}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
