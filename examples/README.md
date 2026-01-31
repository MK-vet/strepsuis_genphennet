# Example Usage Scripts


**Important:** The example datasets are now located in the main repository's data directory at `../../data/`. All CSV files previously stored here have been moved to eliminate duplication.

This directory contains example datasets for testing and learning StrepSuis-GenPhenNet.

## Available Examples

### 1. Basic Example (`basic/`)

**Purpose:** Quick test and learning

**Files included:**
- `AMR_genes.csv`
- `MIC.csv`
- `Virulence.csv`

**Dataset size:** ~91 strains, ~21 features per file

**Expected runtime:** ~1-2 minutes

**What you'll see:**
- Network construction from essential genomic features
- Summary statistics and visualizations
- Interactive HTML reports

**Use case:** First-time users, testing installation

### 2. Advanced Example (`advanced/`)

**Purpose:** Comprehensive analysis with all data types

**Files included:**
- `AMR_genes.csv`
- `MGE.csv`
- `MIC.csv`
- `MLST.csv`
- `Plasmid.csv`
- `Serotype.csv`
- `Virulence.csv`

**Expected runtime:** ~5-8 minutes

**What you'll see:**
- Multi-omics network with community detection
- More detailed associations and patterns
- Complete metadata integration

**Use case:** Publication-ready analysis, exploring all features

## Using These Examples

### Command Line

```bash
# Basic example
strepsuis-genphennet --data-dir ../../data/ --output results_basic/

# Advanced example
strepsuis-genphennet --data-dir ../../data/ --output results_advanced/
```

### Python API

```python
from strepsuis_genphennet import NetworkAnalyzer, Config

# Basic example
config = Config(
    data_dir='../../data/',
    output_dir='results_basic/'
)
analyzer = NetworkAnalyzer(config)
results = analyzer.run()

# Advanced example with custom parameters
config = Config(
    data_dir='../../data/',
    output_dir='results_advanced/',
    fdr_alpha=0.05
)
analyzer = NetworkAnalyzer(config)
results = analyzer.run()
```

### Google Colab
Use the download buttons in the notebook to get these files, or upload them directly.

## Data Format

All example files follow the required format:
- First column: `Strain_ID`
- Binary values: 0 (absent) / 1 (present)
- UTF-8 encoding
- No missing values

## Creating Your Own Data

Use these examples as templates:
1. Keep the same column structure
2. Replace `Strain_ID` values with your strain names
3. Update binary values (0/1) based on your data
4. Ensure no missing values

## Expected Output

Both `basic/` and `advanced/` directories contain `expected_output.txt` files describing what results you should see.

## Questions?

See [USER_GUIDE.md](../USER_GUIDE.md) for detailed data format requirements.
