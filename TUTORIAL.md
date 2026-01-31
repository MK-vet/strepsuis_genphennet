# StrepSuis-GenPhenNet Tutorial

## Quick Start Guide

This tutorial will guide you through using StrepSuis-GenPhenNet for network-based genomic-phenotypic analysis.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Running Analysis](#running-analysis)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)

## Installation

### Option 1: pip install (recommended)

```bash
pip install strepsuis-genphennet
```

### Option 2: From source

```bash
git clone https://github.com/MK-vet/MKrep.git
cd MKrep/separated_repos/strepsuis-genphennet
pip install -e .
```

### Verify installation

```python
import strepsuis_genphennet
print(strepsuis_genphennet.__version__)
```

## Data Preparation

### Required Files

Your data directory should contain:

1. **AMR_genes.csv** - Antimicrobial resistance genes
2. **Virulence.csv** - Virulence factors
3. **MIC.csv** - Phenotypic resistance data

### File Format

All CSV files must have:
- First column: `Strain_ID`
- Binary values: 0 (absent) or 1 (present)
- UTF-8 encoding

Example:
```csv
Strain_ID,Gene1,Gene2,Gene3
Strain001,1,0,1
Strain002,0,1,1
Strain003,1,1,0
```

## Running Analysis

### Command Line Interface

```bash
# Basic usage
strepsuis-genphennet --data-dir ./data --output ./results

# With custom parameters
strepsuis-genphennet \
  --data-dir ./data \
  --output ./results \
  --fdr-alpha 0.05 \
  --min-phi 0.3
```

### Python API

```python
from strepsuis_genphennet import NetworkAnalyzer

# Initialize
analyzer = NetworkAnalyzer(
    data_dir="./data",
    output_dir="./results"
)

# Run analysis
results = analyzer.run()

# Generate reports
analyzer.generate_html_report(results)
analyzer.generate_excel_report(results)
```

## Understanding Results

### Output Files

1. **HTML Report** - Interactive network visualizations
2. **Excel Report** - Association statistics
3. **PNG Charts** - Network graphs, community plots

### Key Metrics

- **Chi-square/Fisher p-values**: Association significance
- **Phi Coefficient**: Association strength
- **FDR-corrected p-values**: Multiple testing correction
- **Network Centrality**: Node importance metrics

## Advanced Usage

### Custom Configuration

```python
from strepsuis_genphennet import Config, NetworkAnalyzer

config = Config(
    fdr_alpha=0.05,
    min_phi=0.3,
    bootstrap_iterations=1000
)

analyzer = NetworkAnalyzer(
    data_dir="./data",
    output_dir="./results",
    config=config
)
```

### Using Innovations

#### Network Motif Analysis

```python
import networkx as nx

# Count triangles
triangles = nx.triangles(G)

# Find feed-forward loops
# (custom implementation in module)
```

#### Causal Discovery

```python
from strepsuis_genphennet.causal_discovery import CausalDiscoveryAnalyzer

# Initialize
causal = CausalDiscoveryAnalyzer(data)

# Build causal graph
graph = causal.build_causal_graph(alpha=0.05)

# Identify causal relationships
edges = causal.get_causal_edges()
```

#### Predictive Modeling

```python
from strepsuis_genphennet.predictive_modeling import PredictiveModeler

# Initialize
modeler = PredictiveModeler(X, y)

# Train models
results = modeler.train_all_models()

# Get best model
best = results['best_model']
print(f"Best AUC: {best['auc']}")
```

## Troubleshooting

### Common Issues

1. **Sparse Network**: Lower min_phi threshold
2. **No Significant Associations**: Check data quality
3. **Memory Error**: Reduce feature count

### Getting Help

- GitHub Issues: https://github.com/MK-vet/strepsuis-genphennet/issues
- Documentation: See README.md and USER_GUIDE.md

## Performance Tips

Based on our benchmarks:

| Operation | Throughput |
|-----------|------------|
| Chi-Square Testing | 691 samples/s |
| Network Construction | 29,666 samples/s |
| Community Detection | 6,518 samples/s |
| Full Pipeline | 1,053 samples/s |

## Network Interpretation

### Centrality Metrics

- **Degree Centrality**: Number of connections
- **Betweenness Centrality**: Bridge nodes
- **Closeness Centrality**: Information spread efficiency

### Community Detection

Uses Louvain algorithm to identify:
- Functionally related gene clusters
- Co-occurring resistance patterns
- Potential horizontal gene transfer events

## Next Steps

- Read [ALGORITHMS.md](ALGORITHMS.md) for details on novel features and algorithms
- See [VALIDATION.md](VALIDATION.md) for statistical validation
- Check [BENCHMARKS.md](BENCHMARKS.md) for performance data
