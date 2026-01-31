"""
Output handler for StrepSuis-GenPhenNet with StandardOutput integration.

This module wraps all output operations to ensure standardized formatting
with statistical interpretation and QA checklists.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import StandardOutput from shared module
try:
    from shared import StandardOutput, create_qa_checklist
except ImportError:
    # Fallback if shared module not in path
    import sys
    shared_path = Path(__file__).parent.parent.parent / "shared" / "src"
    sys.path.insert(0, str(shared_path))
    from shared import StandardOutput, create_qa_checklist


class GenPhenNetOutputHandler:
    """
    Output handler for GenPhenNet network analysis results.

    Wraps all output operations with StandardOutput to ensure consistent
    formatting and inclusion of statistical interpretations and QA checks.
    """

    def __init__(self, output_dir: str):
        """
        Initialize output handler.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_network_topology(
        self,
        topology_df: pd.DataFrame,
        n_nodes: int,
        n_edges: int
    ) -> None:
        """
        Save network topology metrics with interpretation.

        Args:
            topology_df: DataFrame with network topology metrics
            n_nodes: Number of nodes in network
            n_edges: Number of edges in network
        """
        output = StandardOutput(data=topology_df)

        # Calculate network statistics
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0

        interp = f"Network topology analysis of {n_nodes} genes with {n_edges} interactions. "
        interp += f"Network density: {density:.3f} "

        if density > 0.1:
            interp += "(highly connected network indicating strong gene-phenotype associations). "
        elif density > 0.01:
            interp += "(moderately connected network). "
        else:
            interp += "(sparse network with selective interactions). "

        if 'Clustering_Coefficient' in topology_df.columns:
            avg_clustering = topology_df['Clustering_Coefficient'].mean()
            interp += f"Average clustering coefficient: {avg_clustering:.3f}, "
            if avg_clustering > 0.5:
                interp += "indicating strong modular structure."
            else:
                interp += "suggesting hierarchical organization."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Network constructed: {n_nodes} nodes, {n_edges} edges",
            f"✓ Network density: {density:.3f}",
        ]

        if n_edges > 0:
            qa_items.append("✓ Network connectivity verified")
        else:
            qa_items.append("⚠ No edges detected - review correlation threshold")

        output.add_quick_qa(qa_items)
        output.add_metadata("n_nodes", n_nodes)
        output.add_metadata("n_edges", n_edges)
        output.add_metadata("density", density)

        base_path = self.output_dir / "network_topology"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_hub_genes(
        self,
        hub_df: pd.DataFrame,
        degree_threshold: int = 5
    ) -> None:
        """
        Save hub gene analysis with interpretation.

        Args:
            hub_df: DataFrame with hub gene metrics
            degree_threshold: Minimum degree for hub classification
        """
        output = StandardOutput(data=hub_df)

        n_hubs = len(hub_df)

        if n_hubs > 0:
            if 'Degree' in hub_df.columns:
                avg_degree = hub_df['Degree'].mean()
                max_degree = hub_df['Degree'].max()

                interp = f"Identified {n_hubs} hub genes (degree ≥ {degree_threshold}). "
                interp += f"Average hub connectivity: {avg_degree:.1f} interactions. "
                interp += f"Top hub gene shows {max_degree} connections, suggesting central "
                interp += f"regulatory role in gene-phenotype network. "

                if 'Betweenness_Centrality' in hub_df.columns:
                    high_betweenness = (hub_df['Betweenness_Centrality'] > 0.1).sum()
                    interp += f"{high_betweenness} hubs show high betweenness centrality, "
                    interp += f"indicating critical bridging function between network modules."
            else:
                interp = f"Identified {n_hubs} hub genes in the network."
        else:
            interp = f"No hub genes identified with degree threshold ≥ {degree_threshold}."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Hub gene analysis completed (threshold: degree ≥ {degree_threshold})",
        ]

        if n_hubs > 0:
            qa_items.append(f"✓ {n_hubs} hub genes identified")
            if 'Degree' in hub_df.columns:
                qa_items.append(f"✓ Hub connectivity range: {hub_df['Degree'].min()}-{hub_df['Degree'].max()}")
        else:
            qa_items.append("⚠ No hubs found - consider lowering threshold")

        output.add_quick_qa(qa_items)
        output.add_metadata("degree_threshold", degree_threshold)
        output.add_metadata("n_hubs", n_hubs)

        base_path = self.output_dir / "hub_genes"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_community_detection(
        self,
        community_df: pd.DataFrame,
        n_communities: int
    ) -> None:
        """
        Save community detection results with interpretation.

        Args:
            community_df: DataFrame with community assignments
            n_communities: Number of detected communities
        """
        output = StandardOutput(data=community_df)

        # Analyze community structure
        if 'Community' in community_df.columns:
            community_sizes = community_df['Community'].value_counts()
            avg_size = community_sizes.mean()
            largest_community = community_sizes.max()

            interp = f"Community detection identified {n_communities} functional modules. "
            interp += f"Average module size: {avg_size:.1f} genes. "
            interp += f"Largest module contains {largest_community} genes, "
            interp += f"representing a major functional pathway or regulatory network. "

            if 'Modularity' in community_df.columns:
                modularity = community_df['Modularity'].iloc[0] if len(community_df) > 0 else 0
                if modularity > 0.4:
                    interp += f"Modularity score of {modularity:.2f} indicates strong community structure."
                else:
                    interp += f"Modularity score of {modularity:.2f} suggests weak community boundaries."
        else:
            interp = f"Detected {n_communities} communities in the gene-phenotype network."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Community detection completed",
            f"✓ {n_communities} functional modules identified",
        ]

        if 'Community' in community_df.columns:
            qa_items.append(f"✓ Community sizes: {community_sizes.min()}-{community_sizes.max()} genes")

        if 'Modularity' in community_df.columns:
            modularity = community_df['Modularity'].iloc[0] if len(community_df) > 0 else 0
            if modularity > 0.3:
                qa_items.append(f"✓ Good modularity ({modularity:.2f})")
            else:
                qa_items.append(f"⚠ Low modularity ({modularity:.2f})")

        output.add_quick_qa(qa_items)
        output.add_metadata("n_communities", n_communities)

        base_path = self.output_dir / "community_detection"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_gene_phenotype_associations(
        self,
        assoc_df: pd.DataFrame,
        significance_threshold: float = 0.05
    ) -> None:
        """
        Save gene-phenotype associations with interpretation.

        Args:
            assoc_df: DataFrame with association results
            significance_threshold: P-value threshold
        """
        output = StandardOutput(data=assoc_df)

        # Count significant associations
        if 'P_Value' in assoc_df.columns:
            n_significant = (assoc_df['P_Value'] < significance_threshold).sum()
            n_total = len(assoc_df)
            pct_significant = 100 * n_significant / n_total if n_total > 0 else 0

            interp = f"Gene-phenotype association analysis: {n_significant}/{n_total} "
            interp += f"({pct_significant:.1f}%) associations significant (p < {significance_threshold}). "

            if 'Correlation' in assoc_df.columns:
                strong_corr = (abs(assoc_df['Correlation']) > 0.7).sum()
                interp += f"{strong_corr} associations show strong correlation (|r| > 0.7), "
                interp += f"indicating robust gene-phenotype relationships."
        else:
            interp = f"Gene-phenotype association analysis completed for {len(assoc_df)} pairs."

        output.add_statistical_interpretation(interp)

        # QA checklist
        qa_items = [
            f"✓ Association analysis performed on {len(assoc_df)} gene-phenotype pairs",
        ]

        if 'P_Value' in assoc_df.columns:
            if n_significant > 0:
                qa_items.append(f"✓ {n_significant} significant associations (p < {significance_threshold})")
            else:
                qa_items.append("⚠ No significant associations found")

        output.add_quick_qa(qa_items)
        output.add_metadata("significance_threshold", significance_threshold)

        base_path = self.output_dir / "gene_phenotype_associations"
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")

    def save_generic_results(
        self,
        df: pd.DataFrame,
        filename: str,
        interpretation: str,
        qa_items: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save generic results with custom interpretation and QA.

        Args:
            df: DataFrame with results
            filename: Base filename (without extension)
            interpretation: Statistical interpretation text
            qa_items: QA checklist items
            metadata: Optional additional metadata
        """
        output = StandardOutput(data=df)
        output.add_statistical_interpretation(interpretation)
        output.add_quick_qa(qa_items)

        if metadata:
            for key, value in metadata.items():
                output.add_metadata(key, value)

        base_path = self.output_dir / filename
        output.to_json(f"{base_path}.json")
        output.to_csv(f"{base_path}.csv")
        output.to_markdown(f"{base_path}.md")
