#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Coverage Push - Test Main Analysis Components

Target: Push coverage from ~15% to 70%+ by testing:
- perform_full_analysis() with mocked data
- Network construction pipeline
- Report generation (HTML and Excel)
- All utility functions
- Integration with real data
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import itertools

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strepsuis_genphennet import network_analysis_core as nac


@pytest.fixture
def minimal_test_data():
    """Create minimal test dataset."""
    data = {
        'MGE.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'MGE': [f'MGE_{i%3}' for i in range(10)]
        }),
        'MIC.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'Drug_A': np.random.randint(0, 2, 10),
            'Drug_B': np.random.randint(0, 2, 10)
        }),
        'MLST.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'MLST': [f'{i%3}.0' for i in range(10)]
        }),
        'Plasmid.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'Plasmid': [f'P_{i%2}' for i in range(10)]
        }),
        'Serotype.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'Serotype': [f'{i%2}.0' for i in range(10)]
        }),
        'Virulence.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'Vir_1': np.random.randint(0, 2, 10),
            'Vir_2': np.random.randint(0, 2, 10)
        }),
        'AMR_genes.csv': pd.DataFrame({
            'Strain_ID': [f'S{i}' for i in range(10)],
            'AMR_1': np.random.randint(0, 2, 10),
            'AMR_2': np.random.randint(0, 2, 10)
        })
    }
    return data


class TestFullAnalysisPipeline:
    """Test complete analysis workflow."""

    def test_analysis_data_validation(self, minimal_test_data, tmp_path):
        """Test data validation in analysis pipeline."""
        # Write test data
        for filename, df in minimal_test_data.items():
            df.to_csv(tmp_path / filename, index=False)

        # Change to test directory
        original_dir = os.getcwd()
        original_output = nac.output_folder

        try:
            os.chdir(tmp_path)
            nac.output_folder = str(tmp_path / 'output')

            # Mock logging to reduce output
            with patch('logging.info'):
                with patch.object(nac, 'IN_COLAB', False):
                    # Should run successfully
                    nac.perform_full_analysis()

            # Verify output files
            assert (tmp_path / 'output' / 'report.html').exists()
            assert (tmp_path / 'output' / 'network_analysis_log.txt').exists()

        finally:
            os.chdir(original_dir)
            nac.output_folder = original_output

    def test_analysis_with_missing_optional_files(self, tmp_path):
        """Test analysis with only required files."""
        # Create only required files
        required_data = {
            'MGE.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'MGE': ['MGE_1', 'MGE_2', 'MGE_1']
            }),
            'MIC.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'Drug_A': [1, 0, 1]
            }),
            'AMR_genes.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'AMR_1': [1, 1, 0]
            }),
            'Virulence.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'Vir_1': [1, 0, 1]
            }),
            'MLST.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'MLST': ['1', '2', '1']
            }),
            'Plasmid.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'Plasmid': ['P1', 'P2', 'P1']
            }),
            'Serotype.csv': pd.DataFrame({
                'Strain_ID': ['S1', 'S2', 'S3'],
                'Serotype': ['1', '2', '1']
            })
        }

        for filename, df in required_data.items():
            df.to_csv(tmp_path / filename, index=False)

        original_dir = os.getcwd()
        original_output = nac.output_folder

        try:
            os.chdir(tmp_path)
            nac.output_folder = str(tmp_path / 'output')

            with patch('logging.info'), patch('logging.warning'):
                with patch.object(nac, 'IN_COLAB', False):
                    nac.perform_full_analysis()

            # Should complete successfully
            assert (tmp_path / 'output' / 'report.html').exists()

        finally:
            os.chdir(original_dir)
            nac.output_folder = original_output


class TestNetworkVisualization:
    """Test network visualization generation."""

    def test_3d_network_visualization_creation(self):
        """Test 3D network visualization."""
        import networkx as nx
        import plotly.graph_objects as go

        # Create simple network
        G = nx.Graph()
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D')]
        for e in edges:
            G.add_edge(*e, weight=0.8)

        # Create partition
        partition = {'A': 0, 'B': 0, 'C': 1, 'D': 1}

        # Create 3D layout
        pos = nx.spring_layout(G, dim=3, seed=42, k=0.6)

        # Create edge traces
        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color='#888')
        )

        # Create node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        node_colors = ['red' if partition[node] == 0 else 'blue' for node in G.nodes()]

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=10, color=node_colors)
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        assert fig is not None
        assert len(fig.data) == 2


class TestReportGenerationComprehensive:
    """Comprehensive report generation tests."""

    def test_html_report_all_sections(self):
        """Test HTML report with all sections populated."""
        # Create comprehensive test data
        chi2_df = pd.DataFrame({
            'Feature1': ['F1', 'F2', 'F3'],
            'Category1': ['AMR', 'AMR', 'Vir'],
            'Feature2': ['F2', 'F3', 'F4'],
            'Category2': ['AMR', 'Vir', 'MGE'],
            'P_value': [0.001, 0.01, 0.05],
            'Phi_coefficient': [0.9, 0.8, 0.7],
            'CI_Lower': [0.7, 0.6, 0.5],
            'CI_Upper': [0.95, 0.9, 0.85]
        })

        network_df = pd.DataFrame({
            'Feature': ['F1', 'F2', 'F3'],
            'Category': ['AMR', 'AMR', 'Vir'],
            'Cluster': [1, 1, 2],
            'Degree_Centrality': [0.9, 0.7, 0.5],
            'Betweenness_Centrality': [0.8, 0.6, 0.4],
            'Closeness_Centrality': [0.85, 0.65, 0.45],
            'Eigenvector_Centrality': [0.9, 0.7, 0.5]
        })

        entropy_df = pd.DataFrame({
            'Feature1': ['F1', 'F2'],
            'Category1': ['AMR', 'Vir'],
            'Feature2': ['F2', 'F3'],
            'Category2': ['AMR', 'MGE'],
            'Entropy': [1.0, 0.9],
            'Entropy_Normalized': [0.8, 0.7],
            'Conditional_Entropy': [0.5, 0.4],
            'Information_Gain': [0.4, 0.3],
            'Normalized_Mutual_Information': [0.6, 0.5]
        })

        cramers_df = pd.DataFrame({
            'Feature1': ['F1'],
            'Category1': ['AMR'],
            'Feature2': ['F2'],
            'Category2': ['Vir'],
            'Cramers_V': [0.75],
            'CI_Lower': [0.6],
            'CI_Upper': [0.9]
        })

        feature_summary_df = pd.DataFrame({
            'Category': ['AMR', 'Vir', 'MGE'],
            'Number_of_Features': [10, 15, 5]
        })

        excl2_df = pd.DataFrame({
            'Feature_1': ['F1', 'F2'],
            'Category_1': ['AMR', 'Vir'],
            'Feature_2': ['F3', 'F4'],
            'Category_2': ['MGE', 'MLST']
        })

        excl3_df = pd.DataFrame({
            'Feature_1': ['F1'],
            'Category_1': ['AMR'],
            'Feature_2': ['F2'],
            'Category_2': ['Vir'],
            'Feature_3': ['F3'],
            'Category_3': ['MGE']
        })

        hubs_df = pd.DataFrame({
            'Cluster': [1, 1, 2],
            'Feature': ['F1', 'F2', 'F3'],
            'Category': ['AMR', 'AMR', 'Vir'],
            'Degree_Centrality': [0.9, 0.7, 0.5]
        })

        summaries = {
            'chi2': {'Total tests': 100, 'Significant (FDR<0.05)': 20},
            'entropy': {'Mean entropy': 0.8},
            'cramers': {'Mean V': 0.6},
            'excl2': {'Total patterns': 10},
            'excl3': {'Total patterns': 5},
            'network': {'Total nodes': 50, 'Total edges': 120},
            'hubs': {'Total hubs': 15}
        }

        html = nac.generate_report_with_cluster_stats(
            chi2_df, network_df, entropy_df, cramers_df,
            feature_summary_df, excl2_df, excl3_df, hubs_df,
            "<div>Network Visualization</div>",
            summaries['chi2'], summaries['entropy'], summaries['cramers'],
            summaries['excl2'], summaries['excl3'], summaries['network'], summaries['hubs'],
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )

        # Verify all sections present
        assert '<!DOCTYPE html>' in html
        assert 'StrepSuis-GenPhenNet' in html
        assert 'Feature Summary' in html
        assert 'Chi-square Results' in html
        assert 'Information Theory' in html
        assert 'Cramér\'s V' in html
        assert 'Mutually Exclusive Patterns' in html
        assert 'Network Analysis' in html
        assert 'Cluster Hubs' in html
        assert 'DataTable' in html  # Check for DataTable (JavaScript library)

    def test_excel_report_creation(self, tmp_path):
        """Test Excel report creation and structure."""
        nac.output_folder = str(tmp_path)

        # Minimal data
        chi2_df = pd.DataFrame({'Feature1': ['F1'], 'Feature2': ['F2'], 'P_value': [0.01]})
        network_df = pd.DataFrame({'Feature': ['F1'], 'Degree_Centrality': [0.8]})
        feature_summary_df = pd.DataFrame({'Category': ['AMR'], 'Number_of_Features': [10]})

        excel_path = nac.generate_excel_report_with_cluster_stats(
            chi2_df, network_df, pd.DataFrame(), pd.DataFrame(),
            feature_summary_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            None,
            {'Total tests': 10}, {}, {}, {}, {}, {}, {},
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )

        assert os.path.exists(excel_path)
        assert excel_path.endswith('.xlsx')


class TestDataTransformations:
    """Test data transformation functions."""

    def test_feature_category_mapping(self):
        """Test feature to category mapping."""
        data = {
            'AMR_genes.csv': pd.DataFrame({
                'Strain_ID': ['S1'],
                'AMR_1': [1],
                'AMR_2': [1]
            }),
            'MIC.csv': pd.DataFrame({
                'Strain_ID': ['S1'],
                'Drug_A': [8]
            })
        }

        mapping = {}
        for fn in data.keys():
            cat = fn.replace('.csv', '')
            df = data[fn]
            for col in df.columns:
                if col != 'Strain_ID':
                    mapping[col] = cat

        assert mapping['AMR_1'] == 'AMR_genes'
        assert mapping['AMR_2'] == 'AMR_genes'
        assert mapping['Drug_A'] == 'MIC'

    def test_data_merging_workflow(self):
        """Test data merging in analysis workflow."""
        # Create test dataframes
        df1 = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'F1': [1, 0, 1],
            'F2': [0, 1, 1]
        })

        df2 = pd.DataFrame({
            'Strain_ID': ['S1', 'S2', 'S3'],
            'F3': [1, 1, 0],
            'F4': [0, 1, 1]
        })

        # Merge
        merged = df1.merge(df2, on='Strain_ID', how='outer')
        merged.fillna(0, inplace=True)

        assert len(merged) == 3
        assert 'F1' in merged.columns
        assert 'F3' in merged.columns
        assert merged.isnull().sum().sum() == 0


class TestStatisticalPipeline:
    """Test statistical analysis pipeline components."""

    def test_chi_square_batch_processing(self):
        """Test batch processing of chi-square tests."""
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(np.random.binomial(1, 0.5, (30, 5)))
        features = data.columns.tolist()
        pairs = list(itertools.combinations(features, 2))

        results = []
        for f1, f2 in pairs:
            p, phi, _, lo, hi = nac.chi2_phi(data[f1], data[f2])
            results.append({
                'Feature1': f1,
                'Feature2': f2,
                'P_value': p,
                'Phi': phi,
                'CI_Lower': lo,
                'CI_Upper': hi
            })

        results_df = pd.DataFrame(results)

        assert len(results_df) == len(pairs)
        assert all(results_df['P_value'] >= 0) and all(results_df['P_value'] <= 1)

    def test_fdr_correction_pipeline(self):
        """Test FDR correction in pipeline."""
        import statsmodels.stats.multitest as smm

        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        df = pd.DataFrame({'P_value': p_values})

        significant, p_adjusted, _, _ = smm.multipletests(
            df['P_value'],
            method='fdr_bh'
        )

        df['Significant'] = significant
        df['P_adjusted'] = p_adjusted

        assert len(df) == len(p_values)
        assert df['Significant'].iloc[0] == True  # Smallest p-value should be significant
        assert df['Significant'].iloc[-1] == False  # Largest p-value should not be


class TestInformationTheoryPipeline:
    """Test information theory analysis pipeline."""

    def test_entropy_calculation_pipeline(self):
        """Test entropy calculation for multiple features."""
        features = {
            'F1': pd.Series([0, 1, 0, 1] * 10),
            'F2': pd.Series([0, 0, 1, 1] * 10),
            'F3': pd.Series([1, 1, 1, 1] * 10)
        }

        results = []
        for name, series in features.items():
            H, Hn = nac.calculate_entropy(series)
            results.append({
                'Feature': name,
                'Entropy': H,
                'Normalized_Entropy': Hn
            })

        results_df = pd.DataFrame(results)

        assert len(results_df) == 3
        # F3 is constant, should have 0 entropy
        assert results_df[results_df['Feature'] == 'F3']['Entropy'].values[0] == 0.0

    def test_information_gain_matrix(self):
        """Test information gain calculation for feature pairs."""
        data = pd.DataFrame({
            'F1': [0, 0, 1, 1] * 10,
            'F2': [0, 0, 1, 1] * 10,  # Perfectly correlated with F1
            'F3': [0, 1, 0, 1] * 10   # Independent
        })

        # Calculate IG between F1 and F2 (should be high)
        ig_12 = nac.information_gain(data['F1'], data['F2'])

        # Calculate IG between F1 and F3 (should be low)
        ig_13 = nac.information_gain(data['F1'], data['F3'])

        assert ig_12 > ig_13


class TestCausalAndPredictiveIntegration:
    """Test integration of causal discovery and predictive modeling."""

    def test_causal_discovery_integration(self):
        """Test causal discovery integration with main pipeline."""
        from strepsuis_genphennet.causal_discovery import CausalDiscoveryFramework

        # Create test data
        gene_data = pd.DataFrame(
            np.random.binomial(1, 0.5, (30, 5)),
            columns=[f'Gene_{i}' for i in range(5)],
            index=[f'S{i}' for i in range(30)]
        )

        pheno_data = pd.DataFrame(
            np.random.binomial(1, 0.5, (30, 3)),
            columns=[f'Pheno_{i}' for i in range(3)],
            index=[f'S{i}' for i in range(30)]
        )

        associations = pd.DataFrame({
            'Feature1': ['Gene_0', 'Gene_1'],
            'Feature2': ['Pheno_0', 'Pheno_1'],
            'Phi': [0.7, 0.6],
            'FDR_corrected_p': [0.01, 0.02],
            'Significant': [True, True]
        })

        framework = CausalDiscoveryFramework(
            gene_data=gene_data,
            phenotype_data=pheno_data,
            initial_associations=associations
        )

        # Should initialize successfully
        assert framework.gene_data is not None
        assert framework.pheno_data is not None

    def test_predictive_modeling_integration(self):
        """Test predictive modeling integration."""
        from strepsuis_genphennet.predictive_modeling import GenotypePhenotypePredictor

        # Create test data with signal
        np.random.seed(100)
        gene_data = pd.DataFrame(
            np.random.binomial(1, 0.5, (100, 15)),
            columns=[f'Gene_{i}' for i in range(15)],
            index=[f'S{i}' for i in range(100)]
        )

        # Create phenotype correlated with Gene_0
        pheno_data = pd.DataFrame(
            {'Pheno_0': (gene_data['Gene_0'] & np.random.binomial(1, 0.8, 100)).astype(int)},
            index=[f'S{i}' for i in range(100)]
        )

        predictor = GenotypePhenotypePredictor(
            gene_data=gene_data,
            phenotype_data=pheno_data
        )

        # Should initialize successfully
        assert predictor.X is not None
        assert predictor.y is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
