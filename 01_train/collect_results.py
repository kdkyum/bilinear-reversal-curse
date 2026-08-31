"""Collect final eval results for all 27 trained models from the HuggingFace Hub.

Each model folder in kdkyum/gpt_family_relation ships a last_results.json with
its final reverse-relation accuracies. This script aggregates them into
../results/training/aggregated_results.json, which every downstream notebook uses.
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

WEIGHT_DECAYS = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="kdkyum/gpt_family_relation",
                        help="Hub repo with the released models, or a local models directory (e.g. from export_models.py)")
    parser.add_argument("--out", type=str, default="../results/training/aggregated_results.json")
    args = parser.parse_args()

    is_local = os.path.isdir(args.repo_id)
    records = []
    for wd in WEIGHT_DECAYS:
        for seed in SEEDS:
            subfolder = f"lvl3_lr3.0e-04_wd{wd}_wr0.01_s{seed}"
            if is_local:
                path = Path(args.repo_id) / subfolder / "last_results.json"
            else:
                path = hf_hub_download(args.repo_id, f"{subfolder}/last_results.json")
            with open(path) as f:
                records.append(json.load(f))
            print(f"{subfolder}: reverse_uni_acc={records[-1]['reverse_uni_acc']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {out}")


if __name__ == "__main__":
    main()
