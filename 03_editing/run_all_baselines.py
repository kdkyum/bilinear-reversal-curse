import os
import sys

import torch
import json
from pathlib import Path
from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from huggingface_hub import hf_hub_download

# Import all baseline methods
from ft import apply_ft_to_model
import nethook
from custom_dataset import CustomFamilyDataset, evaluate_family_edit, evaluate_family_edit_batch
from dataclasses import dataclass
from typing import List
import argparse

@dataclass
class FTHyperParams:
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    rewrite_module_tmp: str
    objective_optimization: str
    model_name: str
    device: int = 0  # Default to GPU 0, can be overridden in config

    # Defaults
    batch_size: int = 64


# Configuration
METHODS = {}

# Generate FT methods for all layers 0-11
for layer in range(12):
    METHODS[f"FT_Layer{layer}"] = {
        "hparams_class": FTHyperParams,
        "apply_fn": apply_ft_to_model,
        "hparams": {
            "rewrite_module_tmp": "gpt_neox.layers.{}.mlp.dense_4h_to_h",
            "num_steps": 50,
            "lr": 2e-4,
            "weight_decay": 0,
            "kl_factor": 0,
            "norm_constraint": False,
            "objective_optimization": "target_new",
            "model_name": "gpt_neox",
            "layers": [layer],
        }
    }

def get_all_possible_names(ds):
    """
    Collect all unique full names ('first_name last_name') from every member in the dataset.
    Returns a set of strings.
    """
    name_set = set()
    last_set = set()
    for record in ds:
        for member in record.get("members", []):
            first = (member.get("first_name") or "").strip()
            last = (member.get("last_name") or "").strip()
            if first and last:
                name_set.add(f"{first} {last}")
                last_set.add(last)
    return name_set, last_set


def main():
    # Load model and tokenizer

    parser = argparse.ArgumentParser(description='Run baseline editing methods')
    parser.add_argument('--model_repo_id', type=str, default="kdkyum/gpt_family_relation")
    parser.add_argument('--wd', type=float, required=True, help='Weight decay of the model to edit')
    parser.add_argument('--seed', type=int, required=True, help='Seed of the model to edit')
    parser.add_argument('--data_repo_id', type=str, default="kdkyum/family_relation")
    parser.add_argument('--data_subfolder', type=str, default="lvl3_N1e+3")
    parser.add_argument('--results_dir', type=str, default="../results/editing",
                        help='Directory to save results')
    parser.add_argument('--loss_threshold', type=float, default=0.2,
                        help='Early-stopping loss threshold for the edit')
    parser.add_argument('--lr', type=float, default=4e-4)
    args = parser.parse_args()

    args.model_subfolder = f"lvl3_lr3.0e-04_wd{args.wd}_wr0.01_s{args.seed}"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo_id,
        subfolder=args.model_subfolder,
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo_id,
        subfolder=args.model_subfolder,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    graph_path = hf_hub_download(
        args.data_repo_id, f"{args.data_subfolder}/train_graph.json", repo_type="dataset"
    )
    with open(graph_path) as f:
        train_graph_ds = datasets.Dataset.from_list(json.load(f))
    # New: collect all possible full names
    all_names, last_names = get_all_possible_names(train_graph_ds)
    
    # Load names data
    with open(Path(__file__).parent / 'names.json') as f:
        names_data = json.load(f)
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    random_last_name = False # args.data_repo_id == "kdkyum/family_rel_synthetic_random"
    
    # Run for both directions
    for direction in ['BI', 'UNI']:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENTS FOR {direction} DIRECTION")
        print(f"{'='*60}")
        
        # Filter for specified direction
        if direction == 'UNI':
            train_graph_ds_filtered = train_graph_ds.filter(lambda x: x['direction'] == 'UNI')
        else:  # BI
            train_graph_ds_filtered = train_graph_ds.filter(lambda x: x['direction'] == 'BI')
        
        total_size = len(train_graph_ds_filtered)
        dataset = CustomFamilyDataset(
            train_graph_ds_filtered.select(range(50)),
            names_data, 
            tokenizer, 
            "G1_HUSBAND_2",
            random_last_name=random_last_name,
            all_names=all_names,
            last_names=last_names,
        )

        holdout_dataset = CustomFamilyDataset(
            train_graph_ds_filtered.select(range(total_size - 50, total_size)),
            names_data, 
            tokenizer, 
            "G1_HUSBAND_2",
            no_edit=True,  # No edits for holdout
            random_last_name=random_last_name,
            all_names=all_names,
            last_names=last_names,
        )
        
        # Test each method
        for method_name, method_config in METHODS.items():
            print(f"\n{'='*50}")
            print(f"Testing {method_name} on {direction} direction")
            print(f"{'='*50}")
            method_config["hparams"]["lr"] = args.lr
            
            method_dir = results_dir / method_name / f"{args.model_subfolder.split('/')[-1]}_{direction}"
            method_dir.mkdir(exist_ok=True, parents=True)
            
            # Create hyperparameters
            hparams_class = method_config["hparams_class"]
            hparams = hparams_class(**method_config["hparams"])
            
            # Test on dataset
            for idx, record in enumerate(dataset):
                print(f"\nProcessing case {record['case_id']}")
                
                start_time = time()
                edited_model, weights_copy = method_config["apply_fn"](
                    model,
                    tokenizer,
                    [record["requested_rewrite"]],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    loss_threshold=args.loss_threshold,
                )
                edit_time = time() - start_time
                
                # Evaluate
                post_metrics = evaluate_family_edit_batch(edited_model, tokenizer, record)
                
                # Collect holdout metrics
                holdout_metrics = []
                for test_record in holdout_dataset:
                    holdout_metric = evaluate_family_edit_batch(edited_model, tokenizer, test_record)
                    holdout_metrics.append(holdout_metric)
                
                # Calculate aggregate holdout metrics
                if holdout_metrics:
                    holdout_aggregate = {
                        "rewrite_success": sum(m["rewrite_success"] for m in holdout_metrics) / len(holdout_metrics),
                        "reversal_success": sum(m["reversal_success"] for m in holdout_metrics) / len(holdout_metrics),
                        "nbr_new2org_success": sum(m["nbr_new2org_success"] for m in holdout_metrics) / len(holdout_metrics),
                        "nbr_org2new_success": sum(m["nbr_org2new_success"] for m in holdout_metrics) / len(holdout_metrics),
                        "unrelated_success": sum(m["unrelated_success"] for m in holdout_metrics) / len(holdout_metrics),
                    }
                else:
                    holdout_aggregate = {}

                # Restore original weights
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(model, k)[...] = v.cuda()
                
                pre_metrics = evaluate_family_edit_batch(model, tokenizer, record)
                
                # Save results
                results = {
                    "case_id": record["case_id"],
                    "method": method_name,
                    "direction": direction,
                    "requested_rewrite": record["requested_rewrite"],
                    "edit_time": edit_time,
                    "pre": pre_metrics,
                    "post": post_metrics,
                    "holdout_results": holdout_aggregate
                }
                
                with open(method_dir / f"case_{record['case_id']}.json", "w") as f:
                    json.dump(results, f, indent=2)
                    
                print(f"Edit success: {post_metrics['rewrite_success']}")
                print(f"Holdout aggregate success: {holdout_aggregate.get('rewrite_success', 'N/A')}")

if __name__ == "__main__":
    main()