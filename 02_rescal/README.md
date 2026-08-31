# 02: Representation probing and relational algebra (Figures 3 and 4)

Probes the hidden representations of each model for three relational structures (paper §4.2–4.3):

- `run_rescal_experiment.py`: bilinear probe (RESCAL with an SVD-based ridge solver), plus the algebra tests — composition (e.g. M_husband · M_mother ⇒ father) and transpose/inversion (e.g. M_husbandᵀ ⇒ wife).
- `run_translation_experiment.py`: translational probe (relation as a vector offset), plus the negation test.
- `run_lre_jacobian_experiment.py`: Linear Relational Embedding probe (Jacobian-based, following Hernandez et al. 2024), sweeping the scale β over {1.0, …, 5.0}.

Each script handles one model, selected by `--wd` and `--seed`, and writes per-layer, per-relation accuracies as JSON into `../results/`. For example:

```bash
python run_rescal_experiment.py --wd 3.0 --seed 0
```

Models load from the released Hub repo by default; after training your own sweep, pass `--repo-id ../01_train/models` (see `01_train/export_models.py`).

Run all 27 models as a SLURM array (adjust the partition for your cluster):

```bash
sbatch run_sweep.sh
```

For the LRE sample-size sweep from the paper's appendix, pass `--train_size {10,100,500}` and a matching `--results-dir`.

## Figures

Once results exist (and `01_train/collect_results.py` has been run):

- `fig3_probe_comparison.ipynb`: layer-wise accuracy of all three probes, "Reversal Cursed" vs "Not Reversal Cursed" (Figure 3).
- `fig4_bilinear_rescal.ipynb`: composition and transpose accuracy of the fitted relation matrices (Figure 4), plus per-relation bilinear accuracy (appendix).
- `sm_translational.ipynb`, `sm_lre.ipynb`: per-relation appendix figures for the other two probes.

The notebooks keep their stored outputs from the paper as a reference.
