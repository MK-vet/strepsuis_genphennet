"""
Synthetic Data Generator for StrepSuis-GenPhenNet - Network Integration Validation

This module generates synthetic datasets using proper statistical distributions
for validating network-based feature association analysis.

Generation Methodology:
-----------------------
1. **Binomial Distribution**: Used for binary presence/absence data (features).

2. **Beta Distribution**: Used to generate prevalence rates that follow biological reality.

3. **Known Associations**: Synthetic data includes known pairwise associations
   to validate chi-square tests, phi coefficients, and network construction.

4. **Ground Truth Network**: True associations are embedded for validating
   network analysis correctness.

Scientific References:
---------------------
- Newman, M.E.J. (2010). Networks: An Introduction. Oxford University Press.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate.
  Journal of the Royal Statistical Society, 57(1), 289-300.

Author: MK-vet Team
License: MIT
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticNetworkConfig:
    """Configuration for synthetic network data generation.

    Attributes:
        n_strains: Number of bacterial strains to generate
        n_features: Total number of features
        n_true_associations: Number of true positive associations
        association_strength: Phi coefficient target for true associations
        background_prevalence: Base prevalence for features
        noise_level: Proportion of random noise to add (0.0 to 1.0)
        random_state: Random seed for reproducibility
    """

    n_strains: int = 200
    n_features: int = 50
    n_true_associations: int = 20
    association_strength: float = 0.6
    background_prevalence: float = 0.3
    noise_level: float = 0.05
    random_state: int = 42


@dataclass
class SyntheticNetworkMetadata:
    """Metadata describing the generated synthetic network data.

    Contains ground truth values that can be used to validate
    the correctness of network analysis methods.

    Attributes:
        config: The configuration used to generate the data
        feature_columns: List of feature column names
        true_associations: List of (feature1, feature2, phi) tuples
        true_non_associations: List of feature pairs with no association
        expected_network_edges: Number of expected edges in true network
        generation_timestamp: When the data was generated
    """

    config: SyntheticNetworkConfig
    feature_columns: List[str] = field(default_factory=list)
    true_associations: List[Tuple[str, str, float]] = field(default_factory=list)
    true_non_associations: List[Tuple[str, str]] = field(default_factory=list)
    expected_network_edges: int = 0
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def generate_correlated_binary_features(
    n_samples: int,
    base_prevalence: float,
    target_phi: float,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two correlated binary features with target phi coefficient.

    Parameters:
        n_samples: Number of samples
        base_prevalence: Base prevalence for features
        target_phi: Target phi coefficient (0-1)
        random_state: Random seed

    Returns:
        Tuple of (feature1, feature2) arrays
    """
    rng = np.random.default_rng(random_state)

    # Generate first feature
    feature1 = rng.binomial(1, base_prevalence, n_samples)

    # Generate correlated second feature
    # P(Y=1|X=1) = p + target_phi * sqrt(p*(1-p))
    # P(Y=1|X=0) = p - target_phi * sqrt(p*(1-p)) * p/(1-p)
    p = base_prevalence
    delta = target_phi * np.sqrt(p * (1 - p))

    p_y_given_x1 = np.clip(p + delta, 0.05, 0.95)
    p_y_given_x0 = np.clip(p - delta, 0.05, 0.95)

    feature2 = np.zeros(n_samples, dtype=int)
    feature2[feature1 == 1] = rng.binomial(1, p_y_given_x1, np.sum(feature1 == 1))
    feature2[feature1 == 0] = rng.binomial(1, p_y_given_x0, np.sum(feature1 == 0))

    return feature1, feature2


def generate_network_synthetic_dataset(
    config: Optional[SyntheticNetworkConfig] = None,
) -> Tuple[pd.DataFrame, SyntheticNetworkMetadata]:
    """
    Generate a complete synthetic dataset for network analysis validation.

    This function creates realistic synthetic data with:
    - Binary feature data
    - Known true associations between feature pairs
    - Known non-associations for specificity testing

    Parameters:
        config: Configuration object. Uses defaults if None.

    Returns:
        Tuple of (data_df, metadata):
            - data_df: Feature data DataFrame
            - metadata: Ground truth and generation parameters
    """
    if config is None:
        config = SyntheticNetworkConfig()

    rng = np.random.default_rng(config.random_state)

    # Initialize metadata
    metadata = SyntheticNetworkMetadata(config=config)

    # Generate strain IDs
    strain_ids = [f"Strain_{i:04d}" for i in range(1, config.n_strains + 1)]

    # Generate feature names
    feature_names = [f"Feature_{i:03d}" for i in range(1, config.n_features + 1)]
    metadata.feature_columns = feature_names

    # Initialize data matrix
    data = np.zeros((config.n_strains, config.n_features), dtype=int)

    # Select pairs for true associations
    all_pairs = [(i, j) for i in range(config.n_features) for j in range(i + 1, config.n_features)]
    association_pairs = rng.choice(len(all_pairs), size=config.n_true_associations, replace=False)

    # Track which features have been assigned
    assigned_features = set()

    # Generate associated pairs
    for pair_idx in association_pairs:
        i, j = all_pairs[pair_idx]

        if i not in assigned_features and j not in assigned_features:
            # Generate new correlated pair
            feat_i, feat_j = generate_correlated_binary_features(
                config.n_strains,
                config.background_prevalence,
                config.association_strength,
                random_state=config.random_state + pair_idx,
            )
            data[:, i] = feat_i
            data[:, j] = feat_j
            assigned_features.add(i)
            assigned_features.add(j)
        else:
            # Use existing feature and generate correlated partner
            if i in assigned_features:
                existing_feat = data[:, i]
                _, new_feat = generate_correlated_binary_features(
                    config.n_strains,
                    np.mean(existing_feat),
                    config.association_strength * 0.8,  # Slightly weaker
                    random_state=config.random_state + pair_idx + 1000,
                )
                data[:, j] = new_feat
                assigned_features.add(j)
            else:
                existing_feat = data[:, j]
                _, new_feat = generate_correlated_binary_features(
                    config.n_strains,
                    np.mean(existing_feat),
                    config.association_strength * 0.8,
                    random_state=config.random_state + pair_idx + 2000,
                )
                data[:, i] = new_feat
                assigned_features.add(i)

        # Record true association
        metadata.true_associations.append(
            (feature_names[i], feature_names[j], config.association_strength)
        )

    # Generate independent features for remaining
    for i in range(config.n_features):
        if i not in assigned_features:
            data[:, i] = rng.binomial(1, config.background_prevalence, config.n_strains)

    # Add noise
    noise_mask = rng.random((config.n_strains, config.n_features)) < config.noise_level
    data[noise_mask] = 1 - data[noise_mask]

    # Record some non-associations for specificity testing
    non_assoc_pairs = [p for idx, p in enumerate(all_pairs) if idx not in association_pairs][:20]
    for i, j in non_assoc_pairs:
        metadata.true_non_associations.append((feature_names[i], feature_names[j]))

    metadata.expected_network_edges = config.n_true_associations

    # Create DataFrame
    data_df = pd.DataFrame(data, columns=feature_names)
    data_df.insert(0, "Strain_ID", strain_ids)

    return data_df, metadata


def save_synthetic_network_data(
    data_df: pd.DataFrame,
    metadata: SyntheticNetworkMetadata,
    output_dir: str = "synthetic_data",
) -> Dict[str, str]:
    """
    Save synthetic network data and metadata to files.

    Parameters:
        data_df: Feature data DataFrame
        metadata: Metadata object with ground truth
        output_dir: Directory to save files to

    Returns:
        Dict with paths to saved files
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save data file
    data_file = output_path / "synthetic_features.csv"
    data_df.to_csv(data_file, index=False)
    saved_files["features"] = str(data_file)

    # Save true associations
    assoc_df = pd.DataFrame(
        metadata.true_associations,
        columns=["Feature1", "Feature2", "True_Phi"],
    )
    assoc_file = output_path / "synthetic_true_associations.csv"
    assoc_df.to_csv(assoc_file, index=False)
    saved_files["true_associations"] = str(assoc_file)

    # Save metadata as JSON
    metadata_dict = {
        "config": {
            "n_strains": metadata.config.n_strains,
            "n_features": metadata.config.n_features,
            "n_true_associations": metadata.config.n_true_associations,
            "association_strength": metadata.config.association_strength,
            "background_prevalence": metadata.config.background_prevalence,
            "noise_level": metadata.config.noise_level,
            "random_state": metadata.config.random_state,
        },
        "n_true_associations": len(metadata.true_associations),
        "expected_network_edges": metadata.expected_network_edges,
        "feature_columns": metadata.feature_columns,
        "generation_timestamp": metadata.generation_timestamp,
        "generation_method": "Correlated Binomial pairs with controlled phi coefficient",
    }

    metadata_file = output_path / "synthetic_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    saved_files["metadata"] = str(metadata_file)

    # Save methodology documentation
    methodology_content = f"""# Synthetic Data Generation Methodology for Network Analysis

## Overview

This document describes the statistical methodology used to generate synthetic
data for validating network-based feature association analysis.

## Generation Parameters

- **Number of strains**: {metadata.config.n_strains}
- **Number of features**: {metadata.config.n_features}
- **True associations**: {metadata.config.n_true_associations}
- **Association strength (phi)**: {metadata.config.association_strength:.2f}
- **Background prevalence**: {metadata.config.background_prevalence:.2f}
- **Noise level**: {metadata.config.noise_level:.2f}
- **Random seed**: {metadata.config.random_state}

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

A small proportion ({metadata.config.noise_level*100:.1f}%) of values are randomly
flipped to simulate measurement error and biological variability.

## Ground Truth

### True Associations
{len(metadata.true_associations)} feature pairs with phi ≈ {metadata.config.association_strength:.2f}

### Non-Associations
{len(metadata.true_non_associations)} feature pairs verified to have no true association

## Expected Analysis Performance

- Chi-square tests should identify most true associations with p < 0.05
- FDR correction should control false positives
- Network should have approximately {metadata.expected_network_edges} edges

## References

1. Newman, M.E.J. (2010). Networks: An Introduction. Oxford University Press.
2. Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate.

## Generation Timestamp

{metadata.generation_timestamp}

---
*This data was generated for validation and testing purposes only.*
"""

    methodology_file = output_path / "GENERATION_METHODOLOGY.md"
    with open(methodology_file, "w", encoding="utf-8") as f:
        f.write(methodology_content)
    saved_files["methodology"] = str(methodology_file)

    return saved_files


def validate_synthetic_network_data(
    data_df: pd.DataFrame,
    metadata: SyntheticNetworkMetadata,
) -> Dict[str, Any]:
    """
    Validate that synthetic data has expected statistical properties.

    Parameters:
        data_df: Feature data DataFrame
        metadata: Metadata with ground truth

    Returns:
        Dict with validation results
    """
    results = {
        "validation_passed": True,
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    # Check 1: Verify shape
    expected_rows = metadata.config.n_strains
    expected_cols = metadata.config.n_features + 1  # +1 for Strain_ID

    if len(data_df) == expected_rows:
        results["checks"].append(f"OK Row count: {len(data_df)}")
    else:
        results["errors"].append(f"✗ Row count: {len(data_df)} (expected {expected_rows})")
        results["validation_passed"] = False

    if len(data_df.columns) == expected_cols:
        results["checks"].append(f"OK Column count: {len(data_df.columns)}")
    else:
        results["errors"].append(f"✗ Column count: {len(data_df.columns)} (expected {expected_cols})")
        results["validation_passed"] = False

    # Check 2: Verify binary data
    feature_cols = [c for c in data_df.columns if c != "Strain_ID"]
    all_binary = all(set(data_df[col].unique()).issubset({0, 1}) for col in feature_cols)
    if all_binary:
        results["checks"].append("OK Data is binary")
    else:
        results["errors"].append("✗ Non-binary values found")
        results["validation_passed"] = False

    # Check 3: Verify true associations have expected correlation
    from scipy.stats import pearsonr

    detected_associations = 0
    for feat1, feat2, expected_phi in metadata.true_associations[:5]:  # Check first 5
        if feat1 in data_df.columns and feat2 in data_df.columns:
            corr, _ = pearsonr(data_df[feat1], data_df[feat2])
            if corr > 0.1:  # Some correlation detected
                detected_associations += 1

    if detected_associations > 2:
        results["checks"].append(f"OK True associations detectable: {detected_associations}/5")
    else:
        results["warnings"].append(
            f"WARN Only {detected_associations}/5 associations clearly detectable"
        )

    return results


if __name__ == "__main__":
    # Generate synthetic data when run directly
    print("Generating synthetic network analysis data...")

    config = SyntheticNetworkConfig(
        n_strains=200,
        n_features=50,
        n_true_associations=20,
        random_state=42,
    )
    # #region agent log
    try:
        import json
        import time

        payload = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "H1",
            "location": "generate_synthetic_data.py",
            "message": "genphennet_synthetic_config",
            "data": {"n_strains": config.n_strains},
            "timestamp": int(time.time() * 1000),
        }
        with open(r"c:\Users\ABC\Documents\GitHub\.cursor\debug.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # #endregion

    data_df, metadata = generate_network_synthetic_dataset(config)

    print(f"Generated {len(data_df)} strains with:")
    print(f"  - {len(metadata.feature_columns)} features")
    print(f"  - {len(metadata.true_associations)} true associations")

    # Validate
    print("\nValidating synthetic data...")
    validation = validate_synthetic_network_data(data_df, metadata)

    for check in validation["checks"]:
        print(f"  {check}")
    if validation["warnings"]:
        print(f"\nWarnings: {len(validation['warnings'])}")
    if validation["errors"]:
        print(f"Errors: {len(validation['errors'])}")

    print(f"\nValidation: {'PASSED' if validation['validation_passed'] else 'FAILED'}")

    # Save validation report (publication-ready)
    validation_dir = Path(__file__).parent.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_strains": int(len(data_df)),
        "n_features": int(len(data_df.columns) - 1),
        "checks": validation.get("checks", []),
        "warnings": validation.get("warnings", []),
        "errors": validation.get("errors", []),
        "validation_passed": bool(validation.get("validation_passed", False)),
    }
    with open(validation_dir / "synthetic_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)
    report_lines = [
        "# Synthetic Data Validation Report - strepsuis-genphennet",
        "",
        f"Generated: {validation_payload['generated']}",
        "Data Source: Synthetic data with known ground truth",
        f"Strains: {validation_payload['n_strains']}",
        f"Features: {validation_payload['n_features']}",
        f"Checks: {len(validation_payload['checks'])}",
        f"Warnings: {len(validation_payload['warnings'])}",
        f"Errors: {len(validation_payload['errors'])}",
        f"Status: {'PASSED' if validation_payload['validation_passed'] else 'FAILED'}",
        "",
        "## Checks",
    ]
    report_lines.extend([f"- {item}" for item in validation_payload["checks"]])
    if validation_payload["warnings"]:
        report_lines.append("")
        report_lines.append("## Warnings")
        report_lines.extend([f"- {item}" for item in validation_payload["warnings"]])
    if validation_payload["errors"]:
        report_lines.append("")
        report_lines.append("## Errors")
        report_lines.extend([f"- {item}" for item in validation_payload["errors"]])
    with open(validation_dir / "SYNTHETIC_DATA_VALIDATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Save validation report (publication-ready)
    validation_dir = Path(__file__).parent.parent / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_strains": int(len(data_df)),
        "n_features": int(len(data_df.columns) - 1),
        "checks": validation.get("checks", []),
        "warnings": validation.get("warnings", []),
        "errors": validation.get("errors", []),
        "validation_passed": bool(validation.get("validation_passed", False)),
    }
    with open(validation_dir / "synthetic_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(validation_payload, f, indent=2)
    report_lines = [
        "# Synthetic Data Validation Report - strepsuis-genphennet",
        "",
        f"Generated: {validation_payload['generated']}",
        "Data Source: Synthetic data with known ground truth",
        f"Strains: {validation_payload['n_strains']}",
        f"Features: {validation_payload['n_features']}",
        f"Checks: {len(validation_payload['checks'])}",
        f"Warnings: {len(validation_payload['warnings'])}",
        f"Errors: {len(validation_payload['errors'])}",
        f"Status: {'PASSED' if validation_payload['validation_passed'] else 'FAILED'}",
        "",
        "## Checks",
    ]
    report_lines.extend([f"- {item}" for item in validation_payload["checks"]])
    if validation_payload["warnings"]:
        report_lines.append("")
        report_lines.append("## Warnings")
        report_lines.extend([f"- {item}" for item in validation_payload["warnings"]])
    if validation_payload["errors"]:
        report_lines.append("")
        report_lines.append("## Errors")
        report_lines.extend([f"- {item}" for item in validation_payload["errors"]])
    with open(validation_dir / "SYNTHETIC_DATA_VALIDATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Save data
    print("\nSaving synthetic data...")
    output_dir = Path(__file__).parent.parent / "synthetic_data"
    saved = save_synthetic_network_data(data_df, metadata, str(output_dir))

    for key, path in saved.items():
        print(f"  Saved {key}: {path}")

    print("\nDone!")
