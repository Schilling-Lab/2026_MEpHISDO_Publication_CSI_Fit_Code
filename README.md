# hypermri 🧲 (CSI fit edition)

**Schilling-AG python module to work with Bruker MRI data, reduced to loading, reconstructing and fitting CSI data.**

This is a slimmed-down copy of hypermri. It contains only what is needed to
load a Bruker study, reconstruct CSI scans (plus RARE scans as anatomical
reference), view them interactively, and fit the spectra with the robust
multi-peak pipeline (`hypermri.analysis.fitting.robust_fit_pipeline`).
Data is handled as **xarray** datasets.

The full package lives at
[gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t).

## Installation

We strictly recommend using **[Anaconda](https://anaconda.org/)** (or Miniconda) to manage dependencies.

**Prerequisites:**
* **Git**: Required to download specific dependencies (`brukerapi`). [Install Git](https://git-scm.com/downloads).

1. **Create (anaconda) environment**
```bash
conda create -n hypermri_dev python=3.10
conda activate hypermri_dev
conda install git
```

2. **Install package** (run in the folder containing `pyproject.toml`)
```bash
pip install -e .
```

## License

MIT License. See [LICENSE](LICENSE) for details.
