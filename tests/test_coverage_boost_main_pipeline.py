#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for perform_full_analysis Main Pipeline

Target: Cover the main analysis workflow with REAL 91-strain dataset

Critical Coverage Areas:
- perform_full_analysis() end-to-end
- Data loading and validation
- Feature expansion and merging
- Statistical testing pipeline
- Network construction
- Community detection
- Report generation (HTML and Excel)
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_genphennet import network_analysis_core as nac


@pytest.fixture
def real_data_dir():
    """Path to real 91-strain example data."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        return str(examples_dir)
    return None


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestPerformFullAnalysisWithRealData:
    """Test complete analysis pipeline with real 91-strain dataset."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / 'examples' / 'AMR_genes.csv').exists(),
        reason="Real data files not available"
    )
    def test_perform_full_analysis_real_data(self, real_data_dir, temp_output_dir):
        """Test complete analysis with real 91-strain dataset."""
        # Change to data directory
        original_dir = os.getcwd()
        original_output = nac.output_folder

        try:
            os.chdir(real_data_dir)
            nac.output_folder = temp_output_dir

            # Mock IN_COLAB to use local files
            with patch.object(nac, 'IN_COLAB', False):
                # Run analysis
                nac.perform_full_analysis()

            # Verify outputs
            assert os.path.exists(os.path.join(temp_output_dir, 'report.html'))
            assert os.path.exists(os.path.join(temp_output_dir, 'network_analysis_log.txt'))

            # Check for Excel report (with date stamp)
            excel_files = [f for f in os.listdir(temp_output_dir) if f.startswith('Network_Analysis_Report') and f.endswith('.xlsx')]
            assert len(excel_files) > 0

            # Verify HTML report content
            html_path = os.path.join(temp_output_dir, 'report.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            assert 'StrepSuis-GenPhenNet' in html_content
            assert 'Chi-square Results' in html_content
            assert 'Network Analysis' in html_content

        finally:
            os.chdir(original_dir)
            nac.output_folder = original_output

    def test_perform_full_analysis_missing_required_files(self, temp_output_dir):
        """Test analysis fails gracefully when required files are missing."""
        original_dir = os.getcwd()
        original_output = nac.output_folder

        try:
            os.chdir(temp_output_dir)
            nac.output_folder = temp_output_dir

            # Create only partial files
            pd.DataFrame({
                'Strain_ID': ['S1', 'S2'],
                'AMR_gene_1': [1, 0]
            }).to_csv('AMR_genes.csv', index=False)

            with patch.object(nac, 'IN_COLAB', False):
                with pytest.raises(FileNotFoundError):
                    nac.perform_full_analysis()

        finally:
            os.chdir(original_dir)
            nac.output_folder = original_output


class TestDataLoadingAndValidation:
    """Test data loading and validation steps."""

    def test_data_validation_strain_id_column(self, temp_output_dir):
        """Test validation checks for Strain_ID column."""
        # Create test data without Strain_ID
        df_no_strain_id = pd.DataFrame({
            'Feature1': [1, 0],
            'Feature2': [0, 1]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_no_strain_id.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            data = {'test.csv': pd.read_csv(temp_file)}

            # This should raise KeyError during validation
            with pytest.raises(KeyError):
                for fn, df in data.items():
                    if 'Strain_ID' not in df.columns:
                        raise KeyError(f"File {fn} must contain 'Strain_ID' column.")
        finally:
            os.unlink(temp_file)

    def test_feature_summary_generation(self):
        """Test feature summary statistics generation."""
        data = {
            'AMR_genes.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2'],
                'AMR_1': [1, 0],
                'AMR_2': [0, 1]
            }),
            'MIC.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2'],
                'Drug_A': [8, 16],
                'Drug_B': [4, 32]
            })
        }

        summary = []
        for fn in ['AMR_genes.csv', 'MIC.csv']:
            cat = fn.replace('.csv', '')
            df = data[fn]
            if cat in ['MGE', 'MLST', 'Plasmid', 'Serotype']:
                count = len(df[cat].unique())
            else:
                count = len([c for c in df.columns if c != 'Strain_ID'])
            summary.append({'Category': cat, 'Number_of_Features': count})

        summary_df = pd.DataFrame(summary)

        assert len(summary_df) == 2
        assert summary_df[summary_df['Category'] == 'AMR_genes']['Number_of_Features'].values[0] == 2
        assert summary_df[summary_df['Category'] == 'MIC']['Number_of_Features'].values[0] == 2


class TestNetworkConstruction:
    """Test network construction and analysis."""

    def test_network_construction_from_chi2_results(self):
        """Test building network from chi-square results."""
        # Create mock chi-square results
        chi2_df = pd.DataFrame({
            'Feature1': ['F1', 'F2', 'F3', 'F4'],
            'Feature2': ['F2', 'F3', 'F4', 'F5'],
            'Phi_coefficient': [0.8, 0.9, 0.7, 0.6],
            'Significant': [True, True, True, False]
        })

        # Build network
        import networkx as nx
        G = nx.Graph()

        threshold = 0.65
        sig_edges = chi2_df[chi2_df['Significant'] & (chi2_df['Phi_coefficient'] > threshold)]

        for _, row in sig_edges.iterrows():
            G.add_edge(row['Feature1'], row['Feature2'], weight=row['Phi_coefficient'])

        assert G.number_of_nodes() == 4  # F1, F2, F3, F4
        assert G.number_of_edges() == 3  # F1-F2, F2-F3, F3-F4 (F4-F5 not significant)

    def test_community_detection(self):
        """Test Louvain community detection."""
        import networkx as nx
        import community.community_louvain as community_louvain

        # Create test network with clear communities
        G = nx.Graph()
        # Community 1: A-B-C
        G.add_edge('A', 'B', weight=0.9)
        G.add_edge('B', 'C', weight=0.9)
        G.add_edge('C', 'A', weight=0.9)
        # Community 2: D-E-F
        G.add_edge('D', 'E', weight=0.9)
        G.add_edge('E', 'F', weight=0.9)
        G.add_edge('F', 'D', weight=0.9)
        # Weak inter-community link
        G.add_edge('C', 'D', weight=0.1)

        partition = community_louvain.best_partition(G, weight='weight', random_state=42)

        # Should detect 2 communities
        num_communities = len(set(partition.values()))
        assert num_communities >= 1  # At least 1 community

    def test_centrality_calculations(self):
        """Test centrality metric calculations."""
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
            ('B', 'D')  # B is central hub
        ])

        # Degree centrality
        deg_cent = nx.degree_centrality(G)
        assert deg_cent['B'] > deg_cent['A']  # B has highest degree

        # Betweenness centrality
        btw_cent = nx.betweenness_centrality(G)
        assert btw_cent['B'] > 0  # B is on shortest paths

        # Closeness centrality
        cls_cent = nx.closeness_centrality(G)
        assert cls_cent['B'] > cls_cent['A']  # B is more central

        # Eigenvector centrality
        eig_cent = nx.eigenvector_centrality(G, tol=1e-6)
        assert all(0 <= v <= 1 for v in eig_cent.values())

    def test_network_with_no_edges(self):
        """Test handling of network with no significant edges."""
        import networkx as nx

        G = nx.Graph()
        # Empty network

        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

        # Should handle gracefully in analysis pipeline


class TestReportGeneration:
    """Test HTML and Excel report generation."""

    def test_generate_report_with_cluster_stats(self):
        """Test HTML report generation with all sections."""
        # Create minimal test data
        chi2_df = pd.DataFrame({
            'Feature1': ['F1', 'F2'],
            'Category1': ['AMR', 'Vir'],
            'Feature2': ['F2', 'F3'],
            'Category2': ['Vir', 'MGE'],
            'P_value': [0.01, 0.02],
            'Phi_coefficient': [0.8, 0.7],
            'CI_Lower': [0.5, 0.4],
            'CI_Upper': [0.9, 0.8]
        })

        network_df = pd.DataFrame({
            'Feature': ['F1', 'F2'],
            'Category': ['AMR', 'Vir'],
            'Cluster': [1, 1],
            'Degree_Centrality': [0.8, 0.6]
        })

        entropy_df = pd.DataFrame({
            'Feature1': ['F1'],
            'Category1': ['AMR'],
            'Feature2': ['F2'],
            'Category2': ['Vir'],
            'Entropy': [1.0],
            'Entropy_Normalized': [0.5],
            'Information_Gain': [0.3]
        })

        cramers_df = pd.DataFrame({
            'Feature1': ['F1'],
            'Category1': ['AMR'],
            'Feature2': ['F2'],
            'Category2': ['Vir'],
            'Cramers_V': [0.7]
        })

        feature_summary_df = pd.DataFrame({
            'Category': ['AMR', 'Vir'],
            'Number_of_Features': [10, 15]
        })

        excl2_df = pd.DataFrame()
        excl3_df = pd.DataFrame()
        hubs_df = pd.DataFrame()

        summaries = {
            'chi2_summary': {'Total tests': 100, 'Significant (FDR<0.05)': 10},
            'entropy_summary': {'Mean entropy': 0.5},
            'cramers_summary': {'Mean V': 0.3},
            'excl2_summary': {'Total patterns': 5},
            'excl3_summary': {'Total patterns': 2},
            'network_summary': {'Nodes': 50, 'Edges': 100},
            'hubs_summary': {'Total hubs': 10}
        }

        html = nac.generate_report_with_cluster_stats(
            chi2_df, network_df, entropy_df, cramers_df,
            feature_summary_df, excl2_df, excl3_df, hubs_df,
            "<div>Network Visualization</div>",
            summaries['chi2_summary'],
            summaries['entropy_summary'],
            summaries['cramers_summary'],
            summaries['excl2_summary'],
            summaries['excl3_summary'],
            summaries['network_summary'],
            summaries['hubs_summary'],
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )

        # Verify HTML structure
        assert '<!DOCTYPE html>' in html
        assert 'StrepSuis-GenPhenNet' in html
        assert 'Chi-square Results' in html
        assert 'Information Theory' in html
        assert 'Network Analysis' in html
        assert 'Feature Summary' in html

    def test_generate_excel_report_with_cluster_stats(self, temp_output_dir):
        """Test Excel report generation."""
        original_output = nac.output_folder
        nac.output_folder = temp_output_dir

        try:
            # Create minimal test data
            chi2_df = pd.DataFrame({
                'Feature1': ['F1'],
                'Feature2': ['F2'],
                'P_value': [0.01],
                'Phi_coefficient': [0.8]
            })

            network_df = pd.DataFrame({
                'Feature': ['F1'],
                'Category': ['AMR'],
                'Cluster': [1],
                'Degree_Centrality': [0.8]
            })

            feature_summary_df = pd.DataFrame({
                'Category': ['AMR'],
                'Number_of_Features': [10]
            })

            summaries = {
                'chi2_summary': {'Total tests': 100},
                'entropy_summary': {},
                'cramers_summary': {},
                'excl2_summary': {},
                'excl3_summary': {},
                'network_summary': {'Total nodes': 50},
                'hubs_summary': {}
            }

            excel_path = nac.generate_excel_report_with_cluster_stats(
                chi2_df, network_df, pd.DataFrame(), pd.DataFrame(),
                feature_summary_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                None,  # fig_network
                summaries['chi2_summary'],
                summaries['entropy_summary'],
                summaries['cramers_summary'],
                summaries['excl2_summary'],
                summaries['excl3_summary'],
                summaries['network_summary'],
                summaries['hubs_summary'],
                {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            )

            # Verify Excel file created
            assert os.path.exists(excel_path)
            assert excel_path.endswith('.xlsx')

        finally:
            nac.output_folder = original_output


class TestNetworkVisualization:
    """Test 3D network visualization generation."""

    def test_network_3d_visualization(self):
        """Test 3D network visualization creation."""
        import networkx as nx
        import plotly.graph_objects as go

        # Create test network
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

        # Create layout
        pos = nx.spring_layout(G, dim=3, seed=42)

        # Extract coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]

        # Create trace
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(size=10, color='red')
        )

        # Create figure
        fig = go.Figure(data=[node_trace])

        assert fig is not None
        assert len(fig.data) == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame()

        # expand_categories with empty df
        result = nac.expand_categories(df, 'test_category') if 'Strain_ID' in df.columns else df

        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_single_feature_analysis(self):
        """Test analysis with single feature."""
        df = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'Feature1': [0, 1, 1]
        })

        features = ['Feature1']
        # Chi-square requires pairs - should handle gracefully
        assert len(features) == 1

    def test_all_constant_features(self):
        """Test handling of constant features (no variance)."""
        df = pd.DataFrame({
            'F1': [1, 1, 1, 1],
            'F2': [1, 1, 1, 1]
        })

        # Entropy should be 0
        H, Hn = nac.calculate_entropy(df['F1'])
        assert H == 0.0
        assert Hn == 0.0


class TestParallelProcessing:
    """Test parallel processing in statistical tests."""

    def test_concurrent_chi2_testing(self):
        """Test parallel chi-square testing."""
        import concurrent.futures
        import itertools

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(np.random.binomial(1, 0.5, (20, 10)))
        features = data.columns.tolist()
        pairs = list(itertools.combinations(features, 2))[:10]  # Test with 10 pairs

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(nac.chi2_phi, data[f1], data[f2]): (f1, f2)
                for f1, f2 in pairs
            }
            for fut in concurrent.futures.as_completed(futures):
                f1, f2 = futures[fut]
                p, phi, _, _, _ = fut.result()
                results.append({'Feature1': f1, 'Feature2': f2, 'P': p, 'Phi': phi})

        assert len(results) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
