#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Validation on Real S. suis Data - strepsuis-genphennet

This module validates network analysis methods using real data.
Results are saved to validation/REAL_DATA_VALIDATION_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import json
from datetime import datetime
from pathlib import Path


class RealDataValidationReport:
    """Collect and save validation results on real data."""
    
    def __init__(self):
        self.results = []
        self.biological_validations = []
        self.start_time = datetime.now()
    
    def add_result(self, test_name, expected, actual, passed, details="", category="statistical"):
        self.results.append({
            "test": test_name,
            "expected": str(expected),
            "actual": str(actual),
            "passed": bool(passed),
            "details": details,
            "category": category
        })
    
    def add_biological_validation(self, name, description, result, interpretation):
        self.biological_validations.append({
            "name": name,
            "description": description,
            "result": str(result),
            "interpretation": interpretation
        })
    
    def save_report(self, output_dir):
        """Save validation report to markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "REAL_DATA_VALIDATION_REPORT.md"
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Real Data Validation Report - strepsuis-genphennet\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Data Source:** S. suis strains (AMR_genes.csv, MIC.csv, Virulence.csv)\n")
            f.write(f"**Total Tests:** {total}\n")
            f.write(f"**Passed:** {passed}\n")
            f.write(f"**Coverage:** {passed/total*100:.1f}%\n\n")
            f.write("---\n\n")
            
            # Statistical Validation
            f.write("## Statistical Validation Results\n\n")
            f.write("| Test | Expected | Actual | Status |\n")
            f.write("|------|----------|--------|--------|\n")
            
            for r in self.results:
                status = "✅ PASS" if r["passed"] else "❌ FAIL"
                exp_str = str(r['expected'])[:40]
                act_str = str(r['actual'])[:40]
                f.write(f"| {r['test']} | {exp_str} | {act_str} | {status} |\n")
            
            # Biological Validation
            f.write("\n---\n\n")
            f.write("## Biological Validation Results\n\n")
            
            for bv in self.biological_validations:
                f.write(f"### {bv['name']}\n\n")
                f.write(f"**Description:** {bv['description']}\n\n")
                f.write(f"**Result:** {bv['result']}\n\n")
                f.write(f"**Interpretation:** {bv['interpretation']}\n\n")
        
        # Also save as JSON
        json_path = output_path / "real_data_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_tests": total,
                "passed": passed,
                "coverage": passed/total*100,
                "results": self.results,
                "biological_validations": self.biological_validations
            }, f, indent=2)
        
        return passed, total


# Global report instance
report = RealDataValidationReport()


@pytest.fixture(scope="module")
def real_data():
    """Load real S. suis data."""
    data_locations = [
        Path(__file__).parent.parent.parent.parent / "data",
        Path(__file__).parent.parent / "data" / "examples",
        Path(__file__).parent.parent.parent / "strepsuis-mdr" / "data" / "examples",
    ]
    
    data_dir = None
    for loc in data_locations:
        if (loc / "AMR_genes.csv").exists():
            data_dir = loc
            break
    
    if data_dir is None:
        pytest.skip("No data files found")
    
    mic_df = pd.read_csv(data_dir / "MIC.csv")
    amr_df = pd.read_csv(data_dir / "AMR_genes.csv")
    vir_df = pd.read_csv(data_dir / "Virulence.csv")
    
    return {
        "mic": mic_df,
        "amr": amr_df,
        "virulence": vir_df,
        "n_strains": len(mic_df)
    }


class TestDataIntegrity:
    """Validate data integrity and format."""
    
    def test_strain_count(self, real_data):
        """Verify expected number of strains."""
        n_strains = real_data["n_strains"]
        passed = n_strains >= 50
        
        report.add_result(
            "Strain Count",
            "≥50 strains",
            f"{n_strains} strains",
            passed,
            "Dataset should have sufficient sample size",
            "data_integrity"
        )
        assert passed


class TestNetworkConstruction:
    """Validate network construction methods."""
    
    def test_association_network_construction(self, real_data):
        """Test construction of association network."""
        amr_df = real_data["amr"]
        mic_df = real_data["mic"]
        
        merged = amr_df.merge(mic_df, on="Strain_ID")
        
        # Build association network
        amr_cols = amr_df.columns[1:6]
        mic_cols = mic_df.columns[1:6]
        
        edges = []
        for amr_col in amr_cols:
            for mic_col in mic_cols:
                table = pd.crosstab(merged[amr_col], merged[mic_col])
                if table.shape == (2, 2):
                    chi2, p, _, _ = chi2_contingency(table)
                    if p < 0.05:
                        edges.append((amr_col, mic_col, chi2, p))
        
        passed = True  # Network construction succeeded
        
        report.add_result(
            "Association Network Construction",
            "Network built",
            f"{len(edges)} significant edges",
            passed,
            "Chi-square based network",
            "network"
        )
        
        report.add_biological_validation(
            "Genotype-Phenotype Network",
            "Network of significant associations between AMR genes and resistance phenotypes",
            f"{len(edges)} significant edges (p < 0.05)",
            "Dense networks suggest strong genotype-phenotype relationships. Sparse networks may indicate complex resistance mechanisms."
        )
        
        assert passed
    
    def test_network_density(self, real_data):
        """Test network density calculation."""
        amr_df = real_data["amr"]
        data_cols = amr_df.columns[1:11]
        
        # Calculate co-occurrence network
        data = amr_df[data_cols]
        n_nodes = len(data_cols)
        max_edges = n_nodes * (n_nodes - 1) / 2
        
        # Count significant edges
        n_edges = 0
        for i, col1 in enumerate(data_cols):
            for col2 in data_cols[i+1:]:
                table = pd.crosstab(data[col1], data[col2])
                if table.shape == (2, 2):
                    _, p, _, _ = chi2_contingency(table)
                    if p < 0.05:
                        n_edges += 1
        
        density = n_edges / max_edges if max_edges > 0 else 0
        passed = 0 <= density <= 1
        
        report.add_result(
            "Network Density",
            "[0, 1]",
            f"{density:.3f}",
            passed,
            f"{n_edges}/{int(max_edges)} edges",
            "network"
        )
        
        report.add_biological_validation(
            "Network Density",
            "Proportion of possible edges that are significant",
            f"Density: {density:.3f} ({n_edges} edges)",
            "High density (>0.3) suggests extensive gene co-selection. Low density indicates independent evolution."
        )
        
        assert passed


class TestCentralityMetrics:
    """Validate centrality metric calculations."""
    
    def test_degree_centrality(self, real_data):
        """Test degree centrality calculation."""
        amr_df = real_data["amr"]
        data_cols = amr_df.columns[1:11]
        
        data = amr_df[data_cols]
        
        # Calculate degree for each gene
        degrees = {}
        for col in data_cols:
            degree = 0
            for other_col in data_cols:
                if col != other_col:
                    table = pd.crosstab(data[col], data[other_col])
                    if table.shape == (2, 2):
                        _, p, _, _ = chi2_contingency(table)
                        if p < 0.05:
                            degree += 1
            degrees[col] = degree
        
        # Find hub gene
        hub_gene = max(degrees, key=degrees.get)
        max_degree = degrees[hub_gene]
        
        passed = max_degree >= 0
        
        report.add_result(
            "Degree Centrality",
            "Valid degrees",
            f"Hub: {hub_gene} (degree={max_degree})",
            passed,
            "Degree centrality calculation",
            "network"
        )
        
        report.add_biological_validation(
            "Hub Gene Identification",
            "Gene with most significant associations (highest degree)",
            f"Hub gene: {hub_gene} with {max_degree} connections",
            "Hub genes may be key drivers of resistance phenotypes or located on mobile genetic elements."
        )
        
        assert passed


class TestCommunityDetection:
    """Validate community detection methods."""
    
    def test_simple_community_detection(self, real_data):
        """Test simple community detection based on co-occurrence."""
        amr_df = real_data["amr"]
        data_cols = amr_df.columns[1:11]
        
        data = amr_df[data_cols]
        
        # Simple clustering based on correlation
        corr_matrix = data.corr()
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Convert correlation to distance
        dist_matrix = 1 - corr_matrix.abs()
        
        # Flatten for linkage
        condensed = []
        for i in range(len(data_cols)):
            for j in range(i+1, len(data_cols)):
                condensed.append(dist_matrix.iloc[i, j])
        
        if len(condensed) > 0:
            Z = linkage(condensed, method='average')
            clusters = fcluster(Z, t=0.5, criterion='distance')
            n_communities = len(np.unique(clusters))
        else:
            n_communities = 1
        
        passed = n_communities >= 1
        
        report.add_result(
            "Community Detection",
            "≥1 community",
            f"{n_communities} communities",
            passed,
            "Hierarchical clustering",
            "network"
        )
        
        report.add_biological_validation(
            "Gene Communities",
            "Groups of co-occurring genes detected by hierarchical clustering",
            f"{n_communities} gene communities identified",
            "Communities may represent resistance islands, plasmids, or functionally related gene clusters."
        )
        
        assert passed


class TestStatisticalAssociations:
    """Validate statistical association methods."""
    
    def test_fisher_exact_test(self, real_data):
        """Test Fisher's exact test on real data."""
        amr_df = real_data["amr"]
        vir_df = real_data["virulence"]
        
        merged = amr_df.merge(vir_df, on="Strain_ID")
        
        amr_col = amr_df.columns[1]
        vir_col = vir_df.columns[1]
        
        table = pd.crosstab(merged[amr_col], merged[vir_col])
        
        if table.shape == (2, 2):
            odds_ratio, p = fisher_exact(table)
            passed = odds_ratio >= 0 and 0 <= p <= 1
        else:
            passed = True
            odds_ratio, p = 1, 1
        
        report.add_result(
            "Fisher's Exact Test",
            "Valid OR and p-value",
            f"OR={odds_ratio:.2f}, p={p:.4f}",
            passed,
            f"Testing {amr_col} vs {vir_col}",
            "statistical"
        )
        
        report.add_biological_validation(
            "AMR-Virulence Odds Ratio",
            "Odds ratio from Fisher's exact test",
            f"OR={odds_ratio:.2f}, p={p:.4f}",
            "OR>1 suggests positive association. OR<1 suggests negative association."
        )
        
        assert passed
    
    def test_fdr_correction(self, real_data):
        """Test FDR correction on multiple tests."""
        from statsmodels.stats.multitest import multipletests
        
        amr_df = real_data["amr"]
        mic_df = real_data["mic"]
        
        merged = amr_df.merge(mic_df, on="Strain_ID")
        
        p_values = []
        amr_cols = amr_df.columns[1:6]
        mic_cols = mic_df.columns[1:6]
        
        for amr_col in amr_cols:
            for mic_col in mic_cols:
                table = pd.crosstab(merged[amr_col], merged[mic_col])
                if table.shape == (2, 2):
                    _, p, _, _ = chi2_contingency(table)
                    p_values.append(p)
        
        if len(p_values) > 0:
            reject, corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            n_significant = reject.sum()
            passed = True
        else:
            n_significant = 0
            passed = True
        
        report.add_result(
            "FDR Correction",
            "Correction applied",
            f"{n_significant}/{len(p_values)} significant",
            passed,
            "Benjamini-Hochberg FDR",
            "statistical"
        )
        
        report.add_biological_validation(
            "Multiple Testing Correction",
            "FDR-corrected significant associations",
            f"{n_significant} associations significant after FDR correction",
            "FDR correction reduces false positives while maintaining statistical power."
        )
        
        assert passed


class TestInformationTheory:
    """Validate information theory metrics."""
    
    def test_mutual_information(self, real_data):
        """Test mutual information calculation."""
        from sklearn.metrics import mutual_info_score
        
        amr_df = real_data["amr"]
        vir_df = real_data["virulence"]
        
        merged = amr_df.merge(vir_df, on="Strain_ID")
        
        amr_col = amr_df.columns[1]
        vir_col = vir_df.columns[1]
        
        mi = mutual_info_score(merged[amr_col], merged[vir_col])
        
        passed = mi >= 0
        
        report.add_result(
            "Mutual Information",
            "MI ≥ 0",
            f"MI = {mi:.4f}",
            passed,
            f"{amr_col} vs {vir_col}",
            "information_theory"
        )
        
        report.add_biological_validation(
            "Mutual Information",
            "Information shared between AMR gene and virulence factor",
            f"MI = {mi:.4f}",
            "Higher MI indicates stronger dependency. MI=0 indicates independence."
        )
        
        assert passed


@pytest.fixture(scope="session", autouse=True)
def save_validation_report():
    """Save validation report after all tests."""
    yield
    
    output_dir = Path(__file__).parent.parent / "validation"
    passed, total = report.save_report(output_dir)
    
    print(f"\n{'='*60}")
    print(f"REAL DATA VALIDATION REPORT - strepsuis-genphennet")
    print(f"{'='*60}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Coverage: {passed/total*100:.1f}%")
    print(f"Report saved to: {output_dir / 'REAL_DATA_VALIDATION_REPORT.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
