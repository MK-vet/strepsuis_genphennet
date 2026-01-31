"""
Large Data Integration for GenPhenNet Module

This module integrates server-side large data pipeline components for
large network analysis and visualization (5k+ nodes).

Features:
- NetworkX server-side layout computation
- Pre-rendered network images (no raw graph data to browser)
- DuckDB-based network queries
- Support for networks with 10k+ nodes

Author: MK-vet
License: MIT
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Import shared large data pipeline components
try:
    from shared.duckdb_handler import DuckDBHandler
    from shared.networkx_server_layout import NetworkXServerLayout, render_large_network, render_network_communities
    from shared.datashader_plots import DatashaderPlots
    LARGE_DATA_AVAILABLE = True
except ImportError:
    LARGE_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LargeDataGenPhenNet:
    """
    Large-scale network processing for genotype-phenotype networks.

    Handles large networks (5k+ nodes) using server-side layout and rendering.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize large data integration.

        Parameters
        ----------
        output_dir : Path, optional
            Directory for output files
        """
        if not LARGE_DATA_AVAILABLE:
            raise ImportError(
                "Large data pipeline not available. "
                "Install with: pip install duckdb matplotlib networkx"
            )

        self.output_dir = output_dir or Path("genphennet_large_data_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_handler = DuckDBHandler()
        self.layout_renderer = NetworkXServerLayout(figsize=(16, 14), dpi=150)

        logger.info(f"Large data GenPhenNet initialized: {self.output_dir}")

    def load_network_edges(
        self,
        edges_path: Union[str, Path],
        table_name: str = "network_edges"
    ) -> Dict[str, Any]:
        """
        Load large network edge list into DuckDB.

        Parameters
        ----------
        edges_path : str or Path
            Path to edge list CSV or Parquet
        table_name : str
            Table name in DuckDB

        Returns
        -------
        dict
            Dataset metadata
        """
        edges_path = Path(edges_path)

        if edges_path.suffix == '.csv':
            metadata = self.db_handler.load_csv(edges_path, table_name)
        elif edges_path.suffix == '.parquet':
            metadata = self.db_handler.load_parquet(edges_path, table_name)
        else:
            raise ValueError(f"Unsupported file format: {edges_path.suffix}")

        logger.info(f"Loaded {metadata['row_count']:,} edges")
        return metadata

    def build_network_from_db(
        self,
        table_name: str = "network_edges",
        weight_threshold: Optional[float] = None,
        edge_type: Optional[str] = None
    ) -> nx.Graph:
        """
        Build NetworkX graph from database.

        Parameters
        ----------
        table_name : str
            Table name
        weight_threshold : float, optional
            Minimum edge weight
        edge_type : str, optional
            Filter by edge type

        Returns
        -------
        nx.Graph
            NetworkX graph
        """
        where_clauses = []

        if weight_threshold is not None:
            where_clauses.append(f"weight >= {weight_threshold}")

        if edge_type:
            where_clauses.append(f"edge_type = '{edge_type}'")

        where = " AND ".join(where_clauses) if where_clauses else None

        # Query edges (all pages)
        query = f"SELECT source, target, weight FROM {table_name}"
        if where:
            query += f" WHERE {where}"

        result = self.db_handler.connection.execute(query).fetchdf()

        # Build graph
        G = nx.Graph()

        for _, row in result.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['weight'])

        logger.info(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def render_large_network(
        self,
        G: nx.Graph,
        layout: str = 'spring',
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """
        Render large network with server-side layout.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
        layout : str
            Layout algorithm
        output_path : Path, optional
            Output image path
        **kwargs
            Additional rendering parameters

        Returns
        -------
        Path
            Path to saved image
        """
        if output_path is None:
            output_path = self.output_dir / f"network_{layout}.png"

        logger.info(f"Rendering {G.number_of_nodes()} node network with {layout} layout")

        return render_large_network(
            G,
            output_path=output_path,
            layout=layout,
            **kwargs
        )

    def detect_and_visualize_communities(
        self,
        G: nx.Graph,
        output_path: Optional[Path] = None,
        algorithm: str = 'louvain'
    ) -> Tuple[Path, Dict[int, int]]:
        """
        Detect communities and visualize with server-side rendering.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
        output_path : Path, optional
            Output image path
        algorithm : str
            Community detection algorithm

        Returns
        -------
        tuple
            (Path to image, community assignments)
        """
        if output_path is None:
            output_path = self.output_dir / "network_communities.png"

        logger.info(f"Detecting communities with {algorithm} algorithm")

        # Detect communities
        if algorithm == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G)
            except ImportError:
                # Fallback to greedy modularity
                from networkx.algorithms import community
                communities_sets = community.greedy_modularity_communities(G)
                communities = {}
                for comm_id, comm_nodes in enumerate(communities_sets):
                    for node in comm_nodes:
                        communities[node] = comm_id
        elif algorithm == 'greedy_modularity':
            from networkx.algorithms import community
            communities_sets = community.greedy_modularity_communities(G)
            communities = {}
            for comm_id, comm_nodes in enumerate(communities_sets):
                for node in comm_nodes:
                    communities[node] = comm_id
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(f"Found {len(set(communities.values()))} communities")

        # Render with communities
        image_path = render_network_communities(
            G,
            communities,
            output_path=output_path
        )

        return image_path, communities

    def export_network_layout(
        self,
        G: nx.Graph,
        layout: str = 'spring',
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export network layout as JSON for custom rendering.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
        layout : str
            Layout algorithm
        output_path : Path, optional
            Output JSON path

        Returns
        -------
        Path
            Path to JSON file
        """
        if output_path is None:
            output_path = self.output_dir / "network_layout.json"

        return self.layout_renderer.export_layout_data(
            G,
            layout=layout,
            output_path=output_path
        )

    def get_network_statistics(
        self,
        G: nx.Graph
    ) -> Dict[str, Any]:
        """
        Calculate network statistics.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph

        Returns
        -------
        dict
            Network statistics
        """
        logger.info("Calculating network statistics")

        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
        }

        # Connected components
        if nx.is_connected(G):
            stats['connected'] = True
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            stats['connected'] = False
            stats['num_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_component_size'] = len(largest_cc)

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)

        return stats

    def export_top_nodes(
        self,
        G: nx.Graph,
        centrality_measure: str = 'degree',
        n_nodes: int = 100,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export top nodes by centrality measure.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph
        centrality_measure : str
            Centrality measure: 'degree', 'betweenness', 'closeness', 'eigenvector'
        n_nodes : int
            Number of top nodes
        output_path : Path, optional
            Output CSV path

        Returns
        -------
        Path
            Path to CSV file
        """
        if output_path is None:
            output_path = self.output_dir / f"top_nodes_{centrality_measure}.csv"

        logger.info(f"Calculating {centrality_measure} centrality")

        if centrality_measure == 'degree':
            centrality = dict(G.degree())
        elif centrality_measure == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif centrality_measure == 'closeness':
            centrality = nx.closeness_centrality(G)
        elif centrality_measure == 'eigenvector':
            centrality = nx.eigenvector_centrality(G, max_iter=100)
        else:
            raise ValueError(f"Unknown centrality measure: {centrality_measure}")

        # Sort and export top nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n_nodes]

        df = pd.DataFrame(sorted_nodes, columns=['node', centrality_measure])
        df.to_csv(output_path, index=False)

        logger.info(f"Exported top {n_nodes} nodes to {output_path}")
        return output_path

    def close(self):
        """Close database connection."""
        self.db_handler.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def process_large_network(
    edges_path: Union[str, Path],
    output_dir: Union[str, Path],
    layout: str = 'spring',
    detect_communities: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to process large network.

    Parameters
    ----------
    edges_path : str or Path
        Path to edge list
    output_dir : str or Path
        Output directory
    layout : str
        Layout algorithm
    detect_communities : bool
        Whether to detect communities

    Returns
    -------
    dict
        Processing results and output paths
    """
    output_dir = Path(output_dir)

    with LargeDataGenPhenNet(output_dir) as processor:
        # Load edges
        metadata = processor.load_network_edges(edges_path)

        # Build network
        G = processor.build_network_from_db()

        results = {
            'metadata': metadata,
            'network_stats': processor.get_network_statistics(G),
            'outputs': {}
        }

        # Render network
        network_viz = processor.render_large_network(
            G,
            layout=layout,
            node_size=50,
            with_labels=False,
            output_path=output_dir / f"network_{layout}.png"
        )
        results['outputs']['network_visualization'] = str(network_viz)

        # Detect communities
        if detect_communities:
            comm_viz, communities = processor.detect_and_visualize_communities(
                G,
                output_path=output_dir / "network_communities.png"
            )
            results['outputs']['community_visualization'] = str(comm_viz)
            results['num_communities'] = len(set(communities.values()))

        # Export layout
        layout_json = processor.export_network_layout(
            G,
            layout=layout,
            output_path=output_dir / "network_layout.json"
        )
        results['outputs']['layout_json'] = str(layout_json)

        # Export top nodes
        top_nodes = processor.export_top_nodes(
            G,
            centrality_measure='degree',
            n_nodes=100,
            output_path=output_dir / "top_nodes.csv"
        )
        results['outputs']['top_nodes'] = str(top_nodes)

        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Large Data Integration - GenPhenNet Module")
    print("=" * 60)

    if not LARGE_DATA_AVAILABLE:
        print("ERROR: Large data pipeline not available")
        print("Install with: pip install duckdb matplotlib networkx")
        exit(1)

    # Create sample large network
    print("\nCreating sample network (5,000 nodes)...")
    np.random.seed(42)

    # Generate scale-free network
    G_sample = nx.barabasi_albert_graph(5000, 3)

    # Export to edge list
    edges = []
    for u, v in G_sample.edges():
        edges.append({
            'source': u,
            'target': v,
            'weight': np.random.uniform(0.5, 1.0),
            'edge_type': np.random.choice(['genetic', 'phenotypic'])
        })

    df_edges = pd.DataFrame(edges)
    test_file = Path("test_network_large.csv")
    df_edges.to_csv(test_file, index=False)
    print(f"Created: {test_file} ({len(edges):,} edges)")

    # Process network
    print("\nProcessing large network...")
    results = process_large_network(
        test_file,
        "test_network_output",
        layout='spring',
        detect_communities=True
    )

    print(f"\n✓ Processing complete")
    print(f"Network: {results['network_stats']['num_nodes']:,} nodes, "
          f"{results['network_stats']['num_edges']:,} edges")
    print(f"Communities: {results.get('num_communities', 'N/A')}")
    print(f"Outputs: {list(results['outputs'].keys())}")
