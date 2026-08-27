# Ablation-model paper figures

This repository contains the development version of the code used to generate
the ablation-model figures for *On the relationship between atmospheric neutral
density and meteor head echo detection height*.

The implementation uses the `KeroSzasz2008` model from
[`ablate`](https://github.com/jvierine/ablate) (imported as `metablate`) and the
MSIS 2.1 atmosphere.

Run the figure generator from the repository root:

    python make_cabmod_figures.py

It writes the following files under `figures/`:

- `meteor_ablation_single_column.pdf`
- `peak_ablation_height-1.20.pdf`
- `peak_ablation_height-1.00.pdf`
- `peak_ablation_height-0.80.pdf`
- `velocity_shift_table.tex`

Install `ablate` into the active Python environment, or place an `ablate`
checkout next to this repository so that `../ablate/src` is available.
