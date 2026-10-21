# Generate the ablation figures

Install/update the normal Python dependencies (current upstream ablate; no
historical checkout or local source path):

```sh
conda run -n base python -m pip install --upgrade -r requirements-figures.txt
```

Then generate everything with no arguments:

```sh
conda run -n base python make_cabmod_figures.py
```

Outputs in `figures/`:

- `meteor_ablation_single_column.pdf` and `.png`: velocity, absolute mass-loss
  rate, temperature, and mass profiles at 11, 15, 20, 32, 53, and 72 km/s.
- `peak_ablation_height-{1.20,1.00,0.80}.pdf` and `.png`: peak-height density
  comparisons at baseline speeds 11, 15, 20, 32, 52, and 72 km/s. The 52/53
  difference is intentional and matches the reference plots.
- `velocity_shift_table.tex` and `cabmod_results.h5`: table plus full
  trajectories, generating source, and installed dependency provenance.

All three entry elevations (70, 45, 20 degrees) are included. The default
integration duration is 30 s to include low-speed heating peaks; a time-limited
track with mass remaining is not a claim of complete ablation. The generator
requires finite trajectories and interior mass-loss peaks. These are Kero-Szasz
metablate calculations, not the multicomponent CABMOD chemistry model despite
the historical script name. Latest upstream numerical results may differ from
historical screenshots; no archived trajectories or old-source patches are used.

Figure dimensions follow the manuscript's 177 mm text width, with embedded
fonts at least 10 pt. The density-figure legend is below the panels to keep
those labels readable at paper width. Keep script provenance in code comments,
PDF metadata and the HDF5 archive only; never render it in figures or captions.
