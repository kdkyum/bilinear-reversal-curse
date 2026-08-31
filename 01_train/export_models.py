"""Convert training runs into the model layout the analysis stages expect.

For each runs/wd{wd}_seed{seed}/ produced by train.py, this writes
{models_dir}/lvl3_lr3.0e-04_wd{wd}_wr0.01_s{seed}/ containing the checkpoint,
the tokenizer, and a last_results.json with the final eval accuracies
(the same layout as the released kdkyum/gpt_family_relation Hub repo).

The analysis scripts in 02_rescal/ and 03_editing/ then accept the models
directory in place of the Hub repo id, e.g.:

    python run_rescal_experiment.py --repo-id ../01_train/models --wd 3.0 --seed 0
"""

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER = "EleutherAI/gpt-neox-20b"


def export_run(run_dir: Path, models_dir: Path, checkpoint: str, tokenizer) -> None:
    config = evals = None
    with open(run_dir / "log.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            if entry["event"] == "config":
                config = entry
            elif entry["event"] == "eval":
                evals = evals or []
                evals.append(entry)
    if config is None or not evals:
        print(f"skipping {run_dir.name}: incomplete log.jsonl")
        return

    wd, seed = config["weight_decay"], config["seed"]
    dest = models_dir / f"lvl3_lr3.0e-04_wd{wd}_wr0.01_s{seed}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_dir / checkpoint, dest)
    tokenizer.save_pretrained(dest)

    last = evals[-1]
    with open(dest / "last_results.json", "w") as f:
        json.dump(
            {
                "reverse_uni_acc": last["eval_reverse_uni_acc"],
                "best_reverse_acc": max(e["eval_reverse_uni_acc"] for e in evals),
                "reverse_bi_acc": last["eval_reverse_bi_acc"],
                "seed": seed,
                "weight_decay": wd,
            },
            f,
            indent=1,
        )
    print(f"{run_dir.name} -> {dest} (reverse_uni_acc={last['eval_reverse_uni_acc']:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--checkpoint", type=str, default="final", choices=["final", "best"])
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    run_dirs = sorted(p for p in Path(args.runs_dir).iterdir() if (p / "log.jsonl").exists())
    if not run_dirs:
        raise SystemExit(f"No runs with log.jsonl found under {args.runs_dir}")
    for run_dir in run_dirs:
        export_run(run_dir, models_dir, args.checkpoint, tokenizer)


if __name__ == "__main__":
    main()
