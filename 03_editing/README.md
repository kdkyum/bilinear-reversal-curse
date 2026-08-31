# 03: Model editing (Figure 5)

Edits a single fact (A, husband, B) → (A, husband, B') per family and measures whether the edit propagates to logically entailed facts (paper §4.4). Editing is layer-wise fine-tuning of one MLP output matrix (`ft.py`, adapted from [EasyEdit](https://github.com/zjunlp/EasyEdit)): Adam, early stop below a loss threshold of 0.2, one edited model per layer (12 per original model), 50 edits each, evaluated on edit success, reverse/neighborhood generalization, and locality.

`run_all_baselines.py` handles one model, selected by `--wd` and `--seed`:

```bash
python run_all_baselines.py --wd 3.0 --seed 0
```

Models load from the released Hub repo by default; after training your own sweep, pass `--model_repo_id ../01_train/models` (see `01_train/export_models.py`).

Run all 27 models as a SLURM array (adjust the partition for your cluster), then aggregate:

```bash
sbatch run_sweep.sh
# after all jobs finish:
python summarize_results.py   # -> ../results/editing/editing_summary_detailed.csv
```

## Figure

`fig5_editing.ipynb` produces Figure 5 (edit success, logical generalization, locality, and the correlation between bilinear probe accuracy and logical generalization) plus the appendix editing details. It needs `../results/editing/editing_summary_detailed.csv`, the RESCAL results from stage 02, and `../results/training/aggregated_results.json` from stage 01. The notebook keeps its stored outputs from the paper as a reference.
