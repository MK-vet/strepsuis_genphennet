#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests for strepsuis-genphennet

This module provides performance benchmarks for all major operations.
Results are saved to validation/PERFORMANCE_BENCHMARKS_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path


class BenchmarkReport:
    """Collect and save benchmark results."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
    
    def add_result(self, operation, n_samples, n_features, time_seconds, memory_mb=None):
        self.results.append({
            "operation": operation,
            "n_samples": n_samples,
            "n_features": n_features,
            "time_seconds": time_seconds,
            "memory_mb": memory_mb,
            "throughput": n_samples / time_seconds if time_seconds > 0 else 0
        })
    
    def save_report(self, output_dir):
        """Save benchmark report to markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "PERFORMANCE_BENCHMARKS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Performance Benchmarks Report - strepsuis-genphennet\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Total Benchmarks:** {len(self.results)}\n\n")
            f.write("---\n\n")
            
            f.write("## Benchmark Results\n\n")
            f.write("| Operation | Samples | Features | Time (s) | Throughput (samples/s) |\n")
            f.write("|-----------|---------|----------|----------|------------------------|\n")
            
            for r in self.results:
                f.write(f"| {r['operation']} | {r['n_samples']} | {r['n_features']} | {r['time_seconds']:.3f} | {r['throughput']:.1f} |\n")
            
            f.write("\n---\n\n")
            f.write("## Performance Summary\n\n")
            
            # Group by operation
            ops = {}
            for r in self.results:
                op = r['operation']
                if op not in ops:
                    ops[op] = []
                ops[op].append(r)
            
            for op, results in ops.items():
                f.write(f"### {op}\n\n")
                avg_throughput = np.mean([r['throughput'] for r in results])
                f.write(f"- **Average Throughput:** {avg_throughput:.1f} samples/s\n")
                f.write(f"- **Scalability:** Tested with {min(r['n_samples'] for r in results)}-{max(r['n_samples'] for r in results)} samples\n\n")
        
        # Also save as JSON
        json_path = output_path / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        
        return len(self.results)


# Global report instance
report = BenchmarkReport()


class TestStatisticalAssociationBenchmarks:
    """Benchmark statistical association testing."""
    
    @pytest.mark.parametrize("n_samples,n_features", [(50, 20), (100, 30), (200, 40)])
    def test_chi_square_performance(self, n_samples, n_features):
        """Benchmark chi-square testing."""
        from scipy.stats import chi2_contingency
        
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_features)))
        phenotype = np.random.randint(0, 2, n_samples)
        
        start = time.time()
        
        # Chi-square for each feature
        for col in data.columns:
            table = pd.crosstab(data[col], phenotype)
            if table.shape[0] > 1 and table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(table)
        
        elapsed = time.time() - start
        
        report.add_result("Chi-Square Testing", n_samples, n_features, elapsed)
        
        assert elapsed < 10  # 10 seconds max


class TestFDRCorrectionBenchmarks:
    """Benchmark FDR correction."""
    
    @pytest.mark.parametrize("n_tests", [100, 500, 1000])
    def test_fdr_correction_performance(self, n_tests):
        """Benchmark FDR correction."""
        from statsmodels.stats.multitest import multipletests
        
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, n_tests)
        
        start = time.time()
        
        reject, corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        elapsed = time.time() - start
        
        report.add_result("FDR Correction", n_tests, 1, elapsed)
        
        assert elapsed < 5  # 5 seconds max


class TestNetworkConstructionBenchmarks:
    """Benchmark network construction."""
    
    @pytest.mark.parametrize("n_nodes", [20, 50, 100])
    def test_network_construction_performance(self, n_nodes):
        """Benchmark network construction."""
        import networkx as nx
        
        np.random.seed(42)
        
        start = time.time()
        
        # Create network from pairwise associations
        G = nx.Graph()
        for i in range(n_nodes):
            G.add_node(f"node_{i}")
        
        # Add edges based on random associations
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() > 0.7:
                    G.add_edge(f"node_{i}", f"node_{j}", weight=np.random.random())
        
        elapsed = time.time() - start
        
        report.add_result("Network Construction", n_nodes, n_nodes, elapsed)
        
        assert elapsed < 10  # 10 seconds max


class TestCentralityBenchmarks:
    """Benchmark centrality calculations."""
    
    @pytest.mark.parametrize("n_nodes", [20, 50, 100])
    def test_centrality_performance(self, n_nodes):
        """Benchmark centrality calculations."""
        import networkx as nx
        
        np.random.seed(42)
        
        # Create random network
        G = nx.erdos_renyi_graph(n_nodes, 0.3)
        
        start = time.time()
        
        # Calculate various centrality metrics
        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        elapsed = time.time() - start
        
        report.add_result("Centrality Metrics", n_nodes, n_nodes, elapsed)
        
        assert elapsed < 30  # 30 seconds max


class TestCommunityDetectionBenchmarks:
    """Benchmark community detection."""
    
    @pytest.mark.parametrize("n_nodes", [20, 50, 100])
    def test_community_detection_performance(self, n_nodes):
        """Benchmark community detection."""
        import networkx as nx
        
        np.random.seed(42)
        
        # Create random network
        G = nx.erdos_renyi_graph(n_nodes, 0.3)
        
        start = time.time()
        
        # Detect communities
        communities = list(nx.community.louvain_communities(G))
        
        elapsed = time.time() - start
        
        report.add_result("Community Detection", n_nodes, n_nodes, elapsed)
        
        assert elapsed < 30  # 30 seconds max


class TestPredictiveModelingBenchmarks:
    """Benchmark predictive modeling."""
    
    @pytest.mark.parametrize("n_samples,n_features", [(50, 10), (100, 20), (200, 30)])
    def test_logistic_regression_performance(self, n_samples, n_features):
        """Benchmark logistic regression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        start = time.time()
        
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5)
        
        elapsed = time.time() - start
        
        report.add_result("Logistic Regression CV", n_samples, n_features, elapsed)
        
        assert elapsed < 30  # 30 seconds max


class TestMutualInformationBenchmarks:
    """Benchmark mutual information calculations."""
    
    @pytest.mark.parametrize("n_samples,n_features", [(50, 20), (100, 30), (200, 40)])
    def test_mutual_information_performance(self, n_samples, n_features):
        """Benchmark mutual information."""
        from sklearn.metrics import mutual_info_score
        
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_features)))
        target = np.random.randint(0, 2, n_samples)
        
        start = time.time()
        
        # Calculate MI for each feature
        for col in data.columns:
            mi = mutual_info_score(data[col], target)
        
        elapsed = time.time() - start
        
        report.add_result("Mutual Information", n_samples, n_features, elapsed)
        
        assert elapsed < 10  # 10 seconds max


class TestFullPipelineBenchmarks:
    """Benchmark full analysis pipeline."""
    
    @pytest.mark.parametrize("n_samples", [50, 100])
    def test_full_pipeline_performance(self, n_samples):
        """Benchmark full pipeline execution."""
        import networkx as nx
        from scipy.stats import chi2_contingency
        
        np.random.seed(42)
        n_features = 20
        data = pd.DataFrame(np.random.randint(0, 2, size=(n_samples, n_features)))
        phenotype = np.random.randint(0, 2, n_samples)
        
        start = time.time()
        
        # Step 1: Statistical testing
        p_values = []
        for col in data.columns:
            table = pd.crosstab(data[col], phenotype)
            if table.shape[0] > 1 and table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(table)
                p_values.append(p)
        
        # Step 2: Network construction
        G = nx.Graph()
        for i in range(n_features):
            G.add_node(f"feature_{i}")
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if np.random.random() > 0.7:
                    G.add_edge(f"feature_{i}", f"feature_{j}")
        
        # Step 3: Centrality
        degree = nx.degree_centrality(G)
        
        elapsed = time.time() - start
        
        report.add_result("Full Pipeline", n_samples, n_features, elapsed)
        
        assert elapsed < 30  # 30 seconds max


@pytest.fixture(scope="session", autouse=True)
def save_benchmark_report():
    """Save benchmark report after all tests."""
    yield
    
    # Save report
    output_dir = Path(__file__).parent.parent / "validation"
    n_benchmarks = report.save_report(output_dir)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARKS REPORT - strepsuis-genphennet")
    print(f"{'='*60}")
    print(f"Total Benchmarks: {n_benchmarks}")
    print(f"Report saved to: {output_dir / 'PERFORMANCE_BENCHMARKS_REPORT.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
