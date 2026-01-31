# Changelog

All notable changes to StrepSuis-GenPhenNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-11-20

### Added
- Initial release of StrepSuis-GenPhenNet
- Network-based genome-phenome integration
- Hybrid network construction (genotype-phenotype edges)
- Community detection for resistance modules
- Centrality analysis for key resistance determinants
- Interactive network visualizations
- Publication-quality network layouts
- Complete test suite with pytest
- Docker container support
- Google Colab notebook
- Comprehensive documentation
- CI/CD workflows
- Pre-commit hooks
- Example datasets

### Changed
- Optimized GitHub Actions workflows to reduce runner minutes
- Docker builds now only run on releases and manual triggers
- Updated mypy configuration to Python 3.9

### Fixed
- Fixed 5 code quality issues identified by ruff linting
- Fixed all bare except clauses with specific exception handling
- Fixed type annotation issues for mypy compliance
- Fixed example data inclusion in repository

### Features
- Network topology analysis
- Module detection with Louvain algorithm
- Path analysis between genotypes and phenotypes
- Statistical significance testing for network edges
- Interactive HTML reports
- High-resolution network visualizations

### Technical Details
- Python 3.8+ support
- NetworkX-based network analysis
- Plotly for interactive visualizations
- Command-line interface and Python API
- Docker containerization

## Project History

This tool was developed as part of the StrepSuis Suite for bacterial genomics research,
with a focus on *Streptococcus suis* but applicable to any bacterial species.

[Unreleased]: https://github.com/MK-vet/strepsuis-genphennet/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MK-vet/strepsuis-genphennet/releases/tag/v1.0.0
