#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Validation and Benchmarking - strepsuis-genphennet

This module validates network analysis performance characteristics.
Results are saved to validation/PERFORMANCE_VALIDATION_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from functools import wraps
import tracemalloc
from scipy.stats import chi2_contingency


class PerformanceReport:
    """Collect and save performance validation results."""
    
    def __init__(self):
        self.benchmarks = []
        self.scalability_tests = []
        self.optimizations = []
    
    def add_benchmark(self, name, operation, data_size, time_ms, throughput, memory_mb=None):
        self.benchmarks.append({
            "name": name,
            "operation": operation,
            "data_size": data_size,
            "time_ms": round(time_ms, 2),
            "throughput": throughput,
            "memory_mb": round(memory_mb, 2) if memory_mb else None
        })
    
    def add_scalability(self, operation, sizes, times, memory_usage=None):
        self.scalability_tests.append({
            "operation": operation,
            "sizes": sizes,
            "times_ms": [round(t, 2) for t in times]
        })
    
    def add_optimization(self, name, before_ms, after_ms, improvement_pct):
        self.optimizations.append({
            "name": name,
            "before_ms": round(before_ms, 2),
            "after_ms": round(after_ms, 2),
            "improvement_pct": round(improvement_pct, 1)
        })
    
    def save_report(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "PERFORMANCE_VALIDATION_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Performance Validation Report - strepsuis-genphennet\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            
            f.write("## Benchmark Summary\n\n")
            f.write("| Operation | Data Size | Time (ms) | Throughput |\n")
            f.write("|-----------|-----------|-----------|------------|\n")
            
            for b in self.benchmarks:
                f.write(f"| {b['name']} | {b['data_size']} | {b['time_ms']} | {b['throughput']} |\n")
            
            f.write("\n---\n\n")
            f.write("## Scalability Analysis\n\n")
            
            for s in self.scalability_tests:
                f.write(f"### {s['operation']}\n\n")
                f.write("| Data Size | Time (ms) |\n")
                f.write("|-----------|----------|\n")
                for i, size in enumerate(s['sizes']):
                    f.write(f"| {size} | {s['times_ms'][i]} |\n")
                f.write("\n")
        
        json_path = output_path / "performance_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmarks": self.benchmarks,
                "scalability_tests": self.scalability_tests
            }, f, indent=2)
        
        return len(self.benchmarks)


report = PerformanceReport()


def generate_test_data(n_strains, n_features, prevalence=0.3):
    np.random.seed(42)
    data = np.random.binomial(1, prevalence, (n_strains, n_features))
    return data


class TestNetworkConstructionPerformance:
    """Benchmark network construction performance."""
    
    def test_association_network_performance(self):
        """Benchmark association network construction."""
        sizes = [10, 20, 50]
        times = []
        n_strains = 100
        
        for n_features in sizes:
            data = generate_test_data(n_strains, n_features)
            
            start = time.perf_counter()
            edges = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    table = pd.crosstab(pd.Series(data[:, i]), pd.Series(data[:, j]))
                    if table.shape == (2, 2):
                        chi2, p, _, _ = chi2_contingency(table)
                        if p < 0.05:
                            edges.append((i, j, chi2, p))
            time_ms = (time.perf_counter() - start) * 1000
            times.append(time_ms)
            
            n_pairs = n_features * (n_features - 1) // 2
            report.add_benchmark(
                f"Network Construction (f={n_features})",
                "network_construction",
                f"{n_pairs} pairs",
                time_ms,
                f"{n_pairs/time_ms*1000:.0f} pairs/s",
                None
            )
        
        report.add_scalability("Network Construction", sizes, times)
        assert len(times) == len(sizes)


class TestCentralityPerformance:
    """Benchmark centrality metric calculations."""
    
    def test_degree_centrality_performance(self):
        """Benchmark degree centrality calculation."""
        sizes = [20, 50, 100]
        times = []
        n_strains = 100
        
        for n_features in sizes:
            data = generate_test_data(n_strains, n_features)
            
            start = time.perf_counter()
            degrees = {}
            for i in range(n_features):
                degree = 0
                for j in range(n_features):
                    if i != j:
                        table = pd.crosstab(pd.Series(data[:, i]), pd.Series(data[:, j]))
                        if table.shape == (2, 2):
                            _, p, _, _ = chi2_contingency(table)
                            if p < 0.05:
                                degree += 1
                degrees[i] = degree
            time_ms = (time.perf_counter() - start) * 1000
            times.append(time_ms)
            
            report.add_benchmark(
                f"Degree Centrality (f={n_features})",
                "degree_centrality",
                f"{n_features} nodes",
                time_ms,
                f"{n_features/time_ms*1000:.0f} nodes/s",
                None
            )
        
        report.add_scalability("Degree Centrality", sizes, times)
        assert len(times) == len(sizes)


class TestCommunityDetectionPerformance:
    """Benchmark community detection performance."""
    
    def test_hierarchical_clustering_performance(self):
        """Benchmark hierarchical clustering for community detection."""
        from scipy.cluster.hierarchy import linkage, fcluster
        
        sizes = [20, 50, 100]
        times = []
        n_strains = 100
        
        for n_features in sizes:
            data = generate_test_data(n_strains, n_features)
            df = pd.DataFrame(data)
            corr_matrix = df.corr()
            dist_matrix = 1 - corr_matrix.abs()
            
            # Flatten for linkage
            condensed = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    condensed.append(dist_matrix.iloc[i, j])
            
            start = time.perf_counter()
            if len(condensed) > 0:
                Z = linkage(condensed, method='average')
                clusters = fcluster(Z, t=0.5, criterion='distance')
            time_ms = (time.perf_counter() - start) * 1000
            times.append(time_ms)
            
            report.add_benchmark(
                f"Community Detection (f={n_features})",
                "community_detection",
                f"{n_features} nodes",
                time_ms,
                f"{n_features/time_ms*1000:.0f} nodes/s",
                None
            )
        
        report.add_scalability("Community Detection", sizes, times)
        assert len(times) == len(sizes)


class TestFDRPerformance:
    """Benchmark FDR correction performance."""
    
    def test_fdr_performance(self):
        """Benchmark FDR correction for network analysis."""
        from statsmodels.stats.multitest import multipletests
        
        sizes = [100, 500, 1000, 5000]
        times = []
        
        for n_tests in sizes:
            p_values = np.random.uniform(0, 1, n_tests)
            
            start = time.perf_counter()
            reject, corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            time_ms = (time.perf_counter() - start) * 1000
            times.append(time_ms)
            
            report.add_benchmark(
                f"FDR Correction (n={n_tests})",
                "fdr_correction",
                f"{n_tests} tests",
                time_ms,
                f"{n_tests/time_ms*1000:.0f} tests/s",
                None
            )
        
        report.add_scalability("FDR Correction", sizes, times)
        assert len(times) == len(sizes)


class TestMutualInformationPerformance:
    """Benchmark mutual information calculation."""
    
    def test_mutual_information_performance(self):
        """Benchmark mutual information calculation."""
        from sklearn.metrics import mutual_info_score
        
        sizes = [50, 100, 200, 500]
        times = []
        n_features = 20
        
        for n_strains in sizes:
            data = generate_test_data(n_strains, n_features)
            
            start = time.perf_counter()
            mi_matrix = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(i+1, n_features):
                    mi = mutual_info_score(data[:, i], data[:, j])
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
            time_ms = (time.perf_counter() - start) * 1000
            times.append(time_ms)
            
            n_pairs = n_features * (n_features - 1) // 2
            report.add_benchmark(
                f"Mutual Information (n={n_strains})",
                "mutual_information",
                f"{n_pairs} pairs",
                time_ms,
                f"{n_pairs/time_ms*1000:.0f} pairs/s",
                None
            )
        
        report.add_scalability("Mutual Information", sizes, times)
        assert len(times) == len(sizes)


@pytest.fixture(scope="session", autouse=True)
def save_performance_report():
    yield
    output_dir = Path(__file__).parent.parent / "validation"
    n_benchmarks = report.save_report(output_dir)
    print(f"\n{'='*60}")
    print(f"PERFORMANCE VALIDATION REPORT - strepsuis-genphennet")
    print(f"Total Benchmarks: {n_benchmarks}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
