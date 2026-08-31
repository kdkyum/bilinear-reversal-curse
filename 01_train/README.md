# 01: Training (Figure 2, right)

Trains GPT-NeoX models from scratch on the [kdkyum/family_relation](https://huggingface.co/datasets/kdkyum/family_relation) dataset. The dataset's `lvl3_N1e+3` config contains 1,000 synthetic families (10 members, 8 relation types): 500 BI families with all 36 facts in both directions, and 500 UNI families with `father`/`mother` withheld. Each training document is one family's facts in random order (~318M tokens per epoch). `eval_reverse_uni` (6,000 prompts, UNI families) is the reversal curse test; `eval_reverse_bi` is a recall sanity check.

## Configuration

From Appendix A of the paper, fixed in `train.py`:

| | |
|---|---|
| Architecture | GPT-NeoX, 12 layers, hidden 896, 16 heads (head dim 56), FFN 3584, non-parallel residual, RoPE (base 10000, rotary pct 0.25), ~206M params |
| Tokenizer | `EleutherAI/gpt-neox-20b` |
| Dropout | 0.1 attention, 0.1 hidden |
| Optimizer | AdamW, lr 3e-4, betas (0.9, 0.95), grad clip 1.0 |
| Schedule | cosine decay, 1% linear warmup, 20 epochs |
| Batch | 64 sequences × 1024 tokens (global) |
| Precision | bf16 autocast, fp32 weights |

## Training

A single run on 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py --weight_decay 3.0 --seed 0 --out_dir runs/wd3.0_seed0
```

The command line exposes only what the paper varies (`--weight_decay`, `--seed`) plus basic knobs. The dataset is downloaded and tokenized in memory at startup (about 10 minutes). One run is about 97,000 steps and takes roughly 10 to 12 hours on 4× A100 (measured at ~0.36 s/step; the per-device batch of 16 fits on 40 GB cards). Each run writes `log.jsonl` plus `best/` and `final/` HuggingFace checkpoints to `--out_dir`.

The paper's full sweep (weight decay in {0, 0.1, 0.5, 1, 2, 3, 4, 5, 6} × seeds {0, 1, 2}) as a SLURM array — adjust the partition for your cluster:

```bash
sbatch train_sweep.sh
```

All runs reach 100% training accuracy. Test accuracy (`eval_reverse_uni`) depends on weight decay and seed: below 1.0 runs stay reversal cursed (well under 40%), and from 1.0 upward outcomes split by seed, with some runs exceeding 98%. The split is genuinely stochastic, so expect the transition point to vary across seeds.

## Results and Figure 2 (right)

```bash
python collect_results.py    # -> ../results/training/aggregated_results.json
python plot_figure2.py       # -> fig2_right_reverse_acc_vs_wd.pdf
```

By default `collect_results.py` reads the released models on the Hub ([kdkyum/gpt_family_relation](https://huggingface.co/kdkyum/gpt_family_relation)); pass `--repo-id models` to use your own runs after exporting. Models below 40% reverse accuracy are "Reversal Cursed" and above 98% "Not Reversal Cursed"; the same thresholds define the model groups in stages 02 and 03.

## Exporting your own runs for the analysis stages

```bash
python export_models.py      # runs/ -> models/lvl3_lr3.0e-04_wd{wd}_wr0.01_s{seed}/
```

This copies each run's checkpoint (`final/` by default, `--checkpoint best` for the best one), adds the tokenizer, and writes a `last_results.json` — the same layout as the Hub repo. The analysis scripts then take `--repo-id ../01_train/models` (or `--model_repo_id` for the editing runner) instead of the Hub id.
