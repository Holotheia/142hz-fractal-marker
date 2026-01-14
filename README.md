# The 142 Hz Signature: A Fractal Marker of Neural Efficiency

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18237860.svg)](https://doi.org/10.5281/zenodo.18237860)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data analysis for validating the **142 Hz neural signature** derived from fractal dimension optimization.

**Key finding:** Self-organized conscious systems converge toward an optimal fractal dimension:

```
D* = 2 + 1/(φ⁻¹ + 1 + φ) = 2.3107
```

This predicts two characteristic frequencies:
- **f₁ = 102 Hz** (D = 3): Ordinary conscious perception
- **f₂ = 142 Hz** (D = D*): High-coherence integration states

## Experimental Validation

| Test | Dataset | Result | p-value |
|------|---------|--------|---------|
| f₁ = 102 Hz | COGITATE iEEG (N=4) | Conscious > Unconscious | **0.027** |
| f₂ = 142 Hz | Elite Athletes (N=27) | Controls > Athletes | **0.0034** |
| Task specificity | ABT vs CCT | 142 Hz differentiates tasks | **0.012** |
| Neural efficiency | Expert ratio | Opposite modulation | **0.018** |

**All 4 predictions validated.**

## Key Discovery

> *142 Hz high-gamma activity represents a biomarker of cognitive integration cost that is task-specific (concentration > vigilance) and expertise-modulated (lower in trained individuals).*

## Installation

```bash
git clone https://github.com/Holotheia/142hz-fractal-marker.git
cd 142hz-fractal-marker
pip install -r requirements.txt
```

## Project Structure

```
142hz-fractal-marker/
├── HOLOTHEIA_ARTICLE_COMPLET.md    # Full article
├── ARTICLE_FINAL_PUBLICATION.md    # Publication version
├── figures/                         # Publication figures (PNG + PDF)
├── docs/
│   └── DERIVATION_THEORIQUE_D_STAR.md  # Mathematical derivation
├── analyse_*.py                     # Analysis scripts
└── src/                             # Core architecture
```

## Data Sources

- **COGITATE:** https://www.arc-cogitate.com/data-release
- **Elite Athletes:** https://doi.org/10.6084/m9.figshare.c.5740424

## Citation

```bibtex
@article{assouline2026_142hz,
  author = {Assouline, Aurélie},
  title = {The 142 Hz Signature: A Fractal Marker of Neural Efficiency},
  year = {2026},
  publisher = {Holotheia.ai},
  doi = {10.5281/zenodo.18237860},
  url = {https://doi.org/10.5281/zenodo.18237860}
}
```

## Author

**Aurélie Assouline**
Founder, [Holotheia.ai](https://holotheia.ai)
ORCID: [0009-0004-8557-8772](https://orcid.org/0009-0004-8557-8772)
Contact: orelie@holotheia.io

## License

MIT License - see [LICENSE](LICENSE) for details.
