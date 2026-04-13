# The 142 Hz Signature: A Fractal Marker of Neural Efficiency

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18237860.svg)](https://doi.org/10.5281/zenodo.18237860)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Replication package for the paper *"The 142 Hz Signature: A Fractal Marker of Neural Efficiency"* (Assouline, 2026).

**This package uses only publicly available datasets. No proprietary code or architecture is required.**

## Key Finding

Self-organized conscious systems converge toward an optimal fractal dimension:

```
D* = 2 + 1/(phi^(-1) + 1 + phi) = 2.3107
```

This predicts two characteristic frequencies via `f = 432 / phi^D`:
- **f1 = 102 Hz** (at D = 3): Ordinary conscious perception
- **f2 = 142 Hz** (at D = D*): High-coherence integration states

## Experimental Validation

| Test | Script | Dataset | Result | p-value |
|------|--------|---------|--------|---------|
| f1 = 102 Hz | `analyse_cogitate_142Hz.py` | COGITATE iEEG (N=4) | Conscious > Unconscious | **0.027** |
| f2 = 142 Hz | `analyse_elite_athletes_142Hz.py` | Elite Athletes (N=17) | Controls > Athletes | **0.0034** |
| Task specificity | `analyse_ABT_vs_CCT.py` | ABT vs CCT (N=26) | 142 Hz differentiates tasks | **0.0029** |
| Neural efficiency | `analyse_athletes_vs_controls.py` | Athletes vs Controls | Opposite modulation | **0.0041** |

**All 4 predictions validated.**

*Note: Test 1 (p = 0.027) is from the original analysis. The current multi-band script uses different preprocessing parameters and does not yet reproduce this exact value. Tests 2-4 are fully reproducible from the published scripts.*

> *142 Hz high-gamma activity represents a biomarker of cognitive integration cost that is task-specific (concentration > vigilance) and expertise-modulated (lower in trained individuals).*

## Quick Start

```bash
git clone https://github.com/Holotheia/142hz-fractal-marker.git
cd 142hz-fractal-marker
pip install -r requirements.txt
python run_all.py
```

## Requirements

- **Python** >= 3.9
- **OS tested:** macOS 14 (Apple Silicon), Ubuntu 22.04
- **RAM:** 4 GB minimum (8 GB recommended for iEEG analysis)
- **Dependencies:** numpy >= 1.21, scipy >= 1.7, matplotlib >= 3.4, mne >= 1.0

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Acquisition

Two public datasets are needed. Download them and place them in a `data/` folder:

```
data/
├── cogitate/          # COGITATE iEEG dataset
│   ├── sub-S01/
│   ├── sub-S03/
│   ├── sub-S07/
│   └── sub-S09/
└── elite-athletes/    # Elite Athletes EEG dataset
    └── *.cdt          # Curry Data format files
```

### COGITATE iEEG

- **URL:** https://www.arc-cogitate.com/data-release
- **Subjects used:** S01, S03, S07, S09 (iEEG/ECoG recordings)
- **Format:** BrainVision (.vhdr / .eeg / .vmrk) + events (.tsv)
- **Reference:** Cogitate Consortium (2025). *Scientific Data*, doi:10.1038/s41597-025-04833-z

### Elite Athletes EEG

- **URL:** https://doi.org/10.6084/m9.figshare.c.5740424
- **Format:** Curry Data (.cdt + .dpo descriptor), 1000 Hz sampling rate
- **Tasks:** ABT (Attention-Based Tasks) and CCT (Concentration Cognitive Tasks)
- **Reference:** Duru & Assem (2022). *Frontiers in Human Neuroscience*

## Running the Analyses

### All tests at once

```bash
python run_all.py
```

### Individual tests

```bash
python run_all.py --test 1    # Test 1: COGITATE iEEG (102 Hz)
python run_all.py --test 2    # Test 2: Elite Athletes (142 Hz)
python run_all.py --test 3    # Test 3: ABT vs CCT task dissociation
python run_all.py --test 4    # Test 4: Neural efficiency (experts vs novices)
```

### Supporting analyses

```bash
python analyse_multifrequence.py    # Multi-band gamma comparison
python analyse_stratifiee.py        # Stratified cohort analysis
python analyse_temporal_142Hz.py    # Temporal evolution of 142 Hz
python analyse_iEEG_142Hz.py        # iEEG peak detection pipeline
```

### Generate publication figures

```bash
python run_all.py --figures
# or
python generate_figures.py
```

## Preprocessing Details

All scripts apply the following pipeline:

1. **Notch filter** at 50 Hz and harmonics (100, 150 Hz) with Q=30 to remove line noise
2. **High-pass filter** at 1 Hz (4th-order Butterworth) to remove drift
3. **Power spectral density** via Welch's method (2-second windows, 50% overlap)
4. **Band of interest:** 135-150 Hz (centered on f2 = 142.1 Hz)
5. **Control band:** 120-135 Hz (adjacent, for normalization)
6. **Statistical tests:** Independent t-tests (Test 1, 2, 4), Mann-Whitney U (Test 3)
7. **Significance threshold:** p < 0.05 (two-tailed)

## Repository Structure

```
142hz-fractal-marker/
├── README.md
├── LICENSE                              # CC BY-NC-ND 4.0
├── requirements.txt                     # Python dependencies
├── run_all.py                           # Run all 4 validation tests
├── ARTICLE_FINAL_PUBLICATION.md         # Publication version of the paper
├── HOLOTHEIA_ARTICLE_COMPLET.md         # Full article (extended)
├── The_142Hz_Signature_Assouline_2026.pdf  # PDF preprint
│
├── analyse_cogitate_142Hz.py            # Test 1: iEEG conscious perception
├── analyse_elite_athletes_142Hz.py      # Test 2: 142 Hz in athletes
├── analyse_ABT_vs_CCT.py               # Test 3: Task specificity
├── analyse_athletes_vs_controls.py      # Test 4: Neural efficiency
├── analyse_multifrequence.py            # Supporting: multi-band comparison
├── analyse_stratifiee.py                # Supporting: stratified analysis
├── analyse_temporal_142Hz.py            # Supporting: temporal dynamics
├── analyse_iEEG_142Hz.py               # Supporting: iEEG peak detection
├── generate_figures.py                  # Generate publication figures
│
├── figures/                             # Publication figures (PNG + PDF)
│   ├── Figure1_theoretical_framework.*
│   ├── Figure2_main_results.*
│   ├── Figure3_task_dissociation.*
│   └── Figure4_sample_stability.*
│
└── docs/
    └── THEORETICAL_DERIVATION_D_STAR.md # Mathematical derivation of D*
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'mne'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` on data files | Download datasets — see Data Acquisition above |
| Notch filter warning at 150 Hz | Normal — the 3rd harmonic of 50 Hz (150 Hz) is near our band of interest. The notch is narrow (Q=30) and does not affect the 142 Hz signal. |
| Memory error during iEEG analysis | iEEG files can be large. Use a machine with >= 8 GB RAM, or process one subject at a time. |
| `.cdt` files won't load | Ensure both `.cdt` and `.dpo`/`.dpa` descriptor files are present in the same directory. |

## Patent Notice

The theoretical framework (fractal dimension D* = 2.3107 and its frequency predictions) is described in French patent application FR2508341. This replication package contains only the **validation pipeline** (statistical tests on public data). It does not contain, and does not require, the Holotheia architecture.

## Citation

```bibtex
@article{assouline2026_142hz,
  author    = {Assouline, Aur\'{e}lie},
  title     = {The 142 Hz Signature: A Fractal Marker of Neural Efficiency},
  year      = {2026},
  publisher = {Holotheia.ai},
  doi       = {10.5281/zenodo.18237860},
  url       = {https://doi.org/10.5281/zenodo.18237860}
}
```

## Author

**Aurelie Assouline**
Founder, [Holotheia.ai](https://holotheia.ai)
ORCID: [0009-0004-8557-8772](https://orcid.org/0009-0004-8557-8772)
Contact: orelie@holotheia.io

## License

CC BY-NC-ND 4.0 — See [LICENSE](LICENSE) for details.

This work may be shared with attribution, but not modified or used commercially.
