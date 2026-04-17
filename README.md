# Raman Spectroscopy Processor v2.1

A desktop GUI application for post-processing Raman spectroscopy data, built with Python and Tkinter.

## Features

### Data Processing Pipeline
1. **Data Import** - Excel (.xlsx/.xls) and CSV file support
2. **Range Filtering** - Adjustable Raman shift bounds (default 400-3300 cm⁻¹)
3. **Cosmic Ray Removal** - MAD-based modified Z-score spike detection
4. **Savitzky-Golay Smoothing** - Configurable polynomial order and window size
5. **Baseline Correction** - 9 algorithms with interactive real-time preview
6. **Normalization** - 5 methods for quantitative spectral comparison
7. **Batch Processing** - Parallel processing via joblib

### Baseline Correction Algorithms

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **airPLS** | Whittaker | λ, diff_order |
| **arPLS** | Whittaker | λ, diff_order |
| **ALS** | Whittaker | λ, p, diff_order |
| **SNIP** | Peak-Clipping | max_half_window, filter_order |
| ATQ | Polynomial | order, threshold |
| STQ | Polynomial | order, threshold |
| AH | Polynomial | order, threshold |
| SH | Polynomial | order, threshold |
| Morphological | Morphological | half_window |

### Normalization Methods

| Method | Description |
|--------|-------------|
| **SNV** | Standard Normal Variate: (y - mean) / std |
| **Vector (L2)** | Unit L2 norm |
| **Area** | Unit integral normalization |
| **Min-Max** | Scale to [0, 1] |
| **Max Peak** | Normalize to max intensity = 1 |

### Multivariate Analysis
- **PCA** - Principal Component Analysis with scores, loadings, and variance plots
- **NMF** - Non-negative Matrix Factorization with NNDSVDA initialization (deterministic, fast convergence)

### Export
- Excel / CSV data export with automatic JSON parameter files
- PCA/NMF results export with metadata sheets

## Screenshot

```
┌─────────────────────────────────────────────────────┐
│ [Import Data] [Import Parameters]                   │
├──────────────┬──────────────────────────────────────┤
│ 1. Data Load │                                      │
│ 2. Pre-proc  │         Spectrum Plot                │
│ 3. Smooth    │      (Interactive Matplotlib)        │
│ 4. Baseline  │                                      │
│ 5. Normalize │──────────────────────────────────────│
│ 6. Export    │ [✓ Raw] [✓ Smooth] [✓ BL] [✓ Final] │
│ 7. Batch/PCA │                                      │
└──────────────┴──────────────────────────────────────┘
```

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
```

### Run

```bash
python main_app.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computation |
| pandas | Data I/O and manipulation |
| scipy | Savitzky-Golay filter |
| scikit-learn | PCA, NMF, StandardScaler |
| pybaselines | airPLS, arPLS, ALS, SNIP, Morphological baseline |
| matplotlib | Plotting and visualization |
| openpyxl | Excel file export |
| joblib | Parallel batch processing |

## Data Format

Input data should be structured as:

|  | Raman Shift 1 | Raman Shift 2 | ... |
|---|---|---|---|
| Sample 1 | intensity | intensity | ... |
| Sample 2 | intensity | intensity | ... |
| ... | ... | ... | ... |

- **Row 1**: Raman shift values (wavenumbers, cm⁻¹)
- **Column 1**: Sample names
- **Data cells**: Intensity values

## Project Structure

```
Raman-processor/
├── main_app.py                 # Main GUI application
├── processing_logic.py         # Data processing engine
├── baseline_corrector_app.py   # Interactive baseline adjustment window
├── pca_result_window.py        # PCA results display
├── nmf_result_window.py        # NMF results display
├── config.json                 # Default parameters and algorithm registry
├── requirements.txt            # Python dependencies
└── README.md
```

## License

MIT
