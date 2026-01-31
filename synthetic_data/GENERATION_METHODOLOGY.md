# Synthetic Data Generation Methodology for Network Analysis

## Overview

This document describes the statistical methodology used to generate synthetic
data for validating network-based feature association analysis.

## Generation Parameters

- **Number of strains**: 200
- **Number of features**: 50
- **True associations**: 20
- **Association strength (phi)**: 0.60
- **Background prevalence**: 0.30
- **Noise level**: 0.05
- **Random seed**: 42

## Statistical Methods Used

### 1. Correlated Binary Features

For each true association, two binary features are generated with a target
phi coefficient using conditional probabilities:

- P(Y=1|X=1) = p + phi × sqrt(p × (1-p))
- P(Y=1|X=0) = p - phi × sqrt(p × (1-p)) × p/(1-p)

Where p is the base prevalence.

### 2. Independent Features

Features not in true associations are generated independently using
Bernoulli trials with probability = background_prevalence.

### 3. Noise Addition

A small proportion (5.0%) of values are randomly
flipped to simulate measurement error and biological variability.

## Ground Truth

### True Associations
20 feature pairs with phi ≈ 0.60

### Non-Associations
20 feature pairs verified to have no true association

## Expected Analysis Performance

- Chi-square tests should identify most true associations with p < 0.05
- FDR correction should control false positives
- Network should have approximately 20 edges

## References

1. Newman, M.E.J. (2010). Networks: An Introduction. Oxford University Press.
2. Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate.

## Generation Timestamp

2026-01-29T02:29:01.542135

---
*This data was generated for validation and testing purposes only.*
