# Raman Spectroscopy Processor v2.4

A desktop GUI application for post-processing Raman spectroscopy data, built with Python and Tkinter. Supports single-spectrum processing, batch pipelines, multivariate analysis, hyperspectral mapping, wavelength calibration, and a headless CLI mode.

> 📖 **사용법 상세 안내는 [USER_GUIDE.md](USER_GUIDE.md) (한국어 설명서)를 참고하세요.**

---

## What's new

- **v2.4** — Hyperspectral Raman mapping window, plugin architecture for baseline algorithms, headless CLI (`cli.py`).
- **v2.3** — Clustering (HCA / K-means / UMAP), MCR-ALS multivariate curve resolution, wavelength calibration with reference peak library.
- **v2.2** — Peak detection & fitting (lmfit Gaussian / Lorentzian / Voigt / PseudoVoigt), PCA diagnostics (Hotelling T² / Q-residuals / Scree plot), derivative spectra (SG 1st / 2nd), QC metrics (SNR, saturation, spike count).
- **v2.1** — 9 baseline algorithms, 5 normalization methods, parallel batch processing, PCA & NMF.

---

## Features

### 1. Data Processing Pipeline
1. **Data Import** — Excel, CSV, TXT, ASC, DAT (auto-detect wide table vs. two-column single spectrum).
2. **Range Filtering** — Adjustable Raman shift bounds (default 400–3300 cm⁻¹).
3. **Cosmic Ray Removal** — MAD-based modified Z-score spike detection.
4. **Savitzky-Golay Smoothing** — Configurable polynomial order and window size.
5. **Baseline Correction** — 9 built-in algorithms + user plugins, with interactive real-time preview.
6. **Normalization** — 5 methods for quantitative spectral comparison.
7. **Derivative Spectra** — 1st / 2nd order SG derivative (optional).
8. **Batch Processing** — Parallel execution via `joblib` with QC flagging.

### 2. Baseline Correction Algorithms

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **airPLS** | Whittaker | λ, diff_order |
| **arPLS** | Whittaker | λ, diff_order |
| **ALS** | Whittaker | λ, p, diff_order |
| **SNIP** | Peak-Clipping | max_half_window, filter_order |
| ATQ / STQ / AH / SH | Polynomial | order, threshold |
| Morphological | Morphological | half_window |
| **Rolling Ball** (plugin) | Morphological | radius |

Additional algorithms can be dropped in as plugins (see [Plugins](#plugins)).

### 3. Normalization Methods

| Method | Description |
|--------|-------------|
| **SNV** | Standard Normal Variate: (y − mean) / std |
| **Vector (L2)** | Unit L2 norm |
| **Area** | Unit integral normalization |
| **Min-Max** | Scale to [0, 1] |
| **Max Peak** | Normalize to max intensity = 1 |

### 4. Peak Detection & Fitting
- Automatic peak finding with prominence/height/distance thresholds.
- Nonlinear fitting with **lmfit** — Gaussian, Lorentzian, Voigt, Pseudo-Voigt.
- Per-peak report: center, amplitude, FWHM, area, R².

### 5. Multivariate Analysis
- **PCA** — scores, loadings, scree plot, **Hotelling's T²** and **Q-residuals** diagnostics with 95% / 99% confidence limits.
- **NMF** — Non-negative Matrix Factorization (NNDSVDA init).
- **MCR-ALS** — Multivariate Curve Resolution with non-negativity constraints via `pymcr`.
- **Clustering** — Hierarchical (HCA with dendrogram), K-means (with silhouette score), UMAP 2D embedding.

### 6. Wavelength Calibration
Built-in reference peak library (silicon, polystyrene, cyclohexane, acetonitrile, ethanol). Polynomial fit (1st–3rd order) of observed vs. reference peaks; re-applies to Raman shift axis.

### 7. Hyperspectral Raman Mapping
- Load X/Y/shift wide-table mapping files → 3D data cube.
- Band-integrated 2D heatmap (trapezoid / sum / max / mean) with selectable colormap.
- Click any pixel to display its spectrum side-by-side.
- Flatten cube → send to main batch pipeline for PCA / NMF / MCR-ALS / clustering.
- Export heatmap to Excel / CSV.

### 8. Quality Control
Per-spectrum QC metrics reported in the batch output: SNR, saturation ratio, cosmic spike count, baseline drift, automatic flag for low-quality spectra.

### 9. Plugins
Drop a `.py` file into `plugins/baseline/` exposing a `register(registry)` callable, and it's auto-discovered at startup. See [`plugins/baseline/README.md`](plugins/baseline/README.md) and `rolling_ball.py` for the contract.

### 10. Headless CLI
Full batch pipeline without the GUI — useful for scripting, automation, and reproducibility.

```bash
python cli.py input.xlsx output.xlsx --range 400 3300 \
    --baseline airpls --normalize snv --cosmic --parallel --pca --nmf 3
```

A sidecar JSON with all parameters is saved next to the output file.

---

## Installation

### Requirements
- Python 3.11+

### Setup

```bash
# Clone the repository
git clone https://github.com/GTAEKIM/Raman-processor.git
cd Raman-processor

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: UMAP clustering
pip install umap-learn
```

### Run

```bash
# GUI
python main_app.py

# Headless CLI
python cli.py --help
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Numerical / data manipulation |
| scikit-learn | PCA, NMF, KMeans, StandardScaler |
| pybaselines | Whittaker / SNIP / morphological baselines |
| lmfit | Peak fitting (Gaussian / Lorentzian / Voigt) |
| pymcr | MCR-ALS multivariate curve resolution |
| matplotlib | Plotting |
| openpyxl | Excel export |
| joblib | Parallel batch processing |
| umap-learn *(optional)* | UMAP 2D clustering visualization |

---

## Supported File Formats

| Extension | Description |
|-----------|-------------|
| `.xlsx` / `.xls` | Excel workbook |
| `.csv` | Comma-separated values |
| `.txt` | Tab / comma / semicolon / space-separated text |
| `.asc` | ASCII export from Renishaw, Horiba, Bruker, etc. |
| `.dat` | Generic numeric data file |

Comment lines starting with `#` or `%` are skipped automatically.

---

## Data Format

### Multi-spectrum table (Type A)
Standard format for multiple samples:

|  | Shift 1 | Shift 2 | ... |
|---|---|---|---|
| Sample 1 | intensity | intensity | ... |
| Sample 2 | intensity | intensity | ... |

- **Row 1**: Raman shift values (wavenumbers, cm⁻¹)
- **Column 1**: Sample names
- **Data cells**: Intensity values

### Two-column single spectrum (Type B)
Automatically detected when the file contains exactly two numeric columns:

```
400.0   123.4
401.0   125.1
402.0   119.8
...
```

The filename stem is used as the sample name.

### Hyperspectral mapping (Type C)
Wide table with leading `X` and `Y` coordinate columns; remaining columns are Raman shifts.

| X | Y | Shift 1 | Shift 2 | ... |
|---|---|---|---|---|
| 0.0 | 0.0 | 123 | 130 | ... |
| 0.5 | 0.0 | 119 | 128 | ... |

---

## Project Structure

```
Raman-processor/
├── main_app.py                 # Main GUI application
├── processing_logic.py         # Core data processing engine
├── cli.py                      # Headless command-line interface
├── baseline_corrector_app.py   # Interactive baseline adjustment window
├── peak_analysis_window.py     # Peak detection / lmfit fitting
├── pca_result_window.py        # PCA scores / loadings / diagnostics
├── nmf_result_window.py        # NMF components / weights
├── clustering_window.py        # HCA / K-means / UMAP
├── mcr_als_window.py           # MCR-ALS window
├── calibration_window.py       # Wavelength calibration
├── mapping_window.py           # Hyperspectral mapping window
├── plugins/
│   └── baseline/               # Drop-in baseline plugins
│       ├── rolling_ball.py
│       └── README.md
├── config.json                 # Default parameters and algorithm registry
├── requirements.txt
├── README.md                   # This file
└── USER_GUIDE.md               # Korean user manual (한국어 설명서)
```

---

## Version History

| Version | Highlights |
|---------|------------|
| v2.4 | Hyperspectral mapping · Plugin architecture · Headless CLI |
| v2.3 | Clustering · MCR-ALS · Wavelength calibration |
| v2.2 | Peak fitting · PCA diagnostics · Derivatives · QC |
| v2.1 | 9 baselines · 5 normalizations · PCA / NMF · Parallel batch |

---

## License

MIT
