# Bilinear representation mitigates reversal curse

Training and analysis code for the paper ["Bilinear representation mitigates reversal curse and enables consistent model editing"](https://arxiv.org/abs/2509.21993) (arXiv:2509.21993). The paper trains GPT-NeoX language models from scratch on a synthetic family knowledge graph and shows that, with enough weight decay, some models learn a bilinear relational structure that lets them answer reverse facts they never saw in training and propagate model edits to logically entailed facts.

The repo is one pipeline, in three numbered stages:

| Stage | What it does | Paper figures |
|---|---|---|
| [`01_train`](01_train/) | Trains the 27-model weight-decay × seed sweep, collects eval results | Figure 2 (right) |
| [`02_rescal`](02_rescal/) | Probes hidden representations (bilinear, translational, linear) and tests relational algebra | Figures 3 and 4 |
| [`03_editing`](03_editing/) | Edits one fact per model and measures logical propagation | Figure 5 |

Released artifacts on the HuggingFace Hub:

- Models: [kdkyum/gpt_family_relation](https://huggingface.co/kdkyum/gpt_family_relation) — the 27 trained models (weight decay in {0, 0.1, 0.5, 1, 2, 3, 4, 5, 6} × seeds {0, 1, 2}).
- Dataset: [kdkyum/family_relation](https://huggingface.co/datasets/kdkyum/family_relation) — the synthetic family knowledge graph. All scripts download it automatically.

## Setup

Requires Python ≥ 3.12 and CUDA GPUs (stage 01 needs 4 for training; 02 and 03 need 1). With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

## Replicating the analysis with the released models

The analysis stages default to the released Hub models, so no training is needed:

```bash
cd 01_train
python collect_results.py    # aggregate the 27 models' final eval accuracies
python plot_figure2.py       # Figure 2 (right)

cd ../02_rescal
sbatch run_sweep.sh          # 27 single-GPU jobs, ~2 h each
# then run fig3_probe_comparison.ipynb and fig4_bilinear_rescal.ipynb

cd ../03_editing
sbatch run_sweep.sh          # 27 single-GPU jobs, ~2 h each
python summarize_results.py  # after all jobs finish
# then run fig5_editing.ipynb
```

Experiment scripts write JSON into the shared `results/` directory; notebooks read from there and save figures next to themselves. Stage 01 must run before the notebooks in 02 and 03 (they use its output to group models into "Reversal Cursed" and "Not Reversal Cursed"), and `fig5_editing.ipynb` also needs the RESCAL results from stage 02. The notebooks are stored with their outputs, so you can compare regenerated plots against the paper's.

## Replicating from scratch

To retrain the models instead of using the released ones:

```bash
cd 01_train
sbatch train_sweep.sh              # 27 jobs, 4 GPUs each, ~10-12 h per run
python export_models.py            # runs/ -> models/ in the analysis-ready layout
python collect_results.py --repo-id models
```

then pass your models directory to the analysis scripts, e.g. `python run_rescal_experiment.py --repo-id ../01_train/models --wd 3.0 --seed 0` (and `--model_repo_id ../01_train/models` for `run_all_baselines.py`). The training configuration is fixed to the paper's Appendix A; see [`01_train/README.md`](01_train/README.md).

## Citation

```bibtex
@inproceedings{
kim2026bilinear,
title={Bilinear representation mitigates reversal curse and enables consistent model editing},
author={Dong-Kyum Kim and Minsung Kim and Jea Kwon and Nakyeong Yang and Meeyoung Cha},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=pdNaYcApbz}
}
```
