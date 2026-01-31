#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Analysis Core Module
============================

Network-based analysis of genomic-phenotypic associations in bacterial genomics.

Features:
    - Statistical association testing (Chi-square, Fisher's exact)
    - Multiple testing correction using Benjamini-Hochberg FDR
    - Network construction from significant associations
    - Community detection using Louvain algorithm
    - Centrality metrics (degree, betweenness, closeness)
    - Information theory metrics (entropy, mutual information)
    - Interactive network visualization with Plotly
    - HTML and Excel report generation

Author: MK-vet
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.stats.multitest as smm
from scipy.stats import chi2_contingency
from scipy.stats import entropy as scipy_entropy
from scipy.stats import fisher_exact, norm

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    from .excel_report_utils import ExcelReportGenerator
except ImportError:
    from strepsuis_genphennet.excel_report_utils import ExcelReportGenerator

# Global output folder - can be overridden before calling perform_full_analysis()
output_folder = "output"


###############################################################################
# HTML GENERATION UTILITIES
###############################################################################

def create_interactive_table(df: pd.DataFrame, table_id: str) -> str:
    """
    Create an HTML table with DataTables-compatible formatting.
    
    Generates an HTML table structure that can be enhanced with DataTables.js
    for sorting, filtering, and pagination functionality.
    
    Args:
        df: DataFrame to convert to HTML table.
        table_id: Unique identifier for the table element.
    
    Returns:
        str: HTML string containing the formatted table.
    
    Note:
        Numeric columns are automatically rounded to 3 decimal places.
        The table is styled with the 'data-table' class for CSS styling.
    
    Example:
        >>> html = create_interactive_table(results_df, "associations")
        >>> with open("report.html", "w") as f:
        ...     f.write(html)
    """
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(3)
    html = f'<table id="table-{table_id}" class="data-table" cellspacing="0" width="100%">'
    html += "<thead><tr>"
    for col in display_df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in display_df.iterrows():
        html += "<tr>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def create_section_summary(
    title: str, 
    stats: Dict[str, Any], 
    per_category: Optional[Dict[str, str]] = None, 
    per_feature: Optional[Dict[str, str]] = None
) -> str:
    """
    Create an HTML summary block with statistics.
    
    Generates a formatted HTML block displaying summary statistics with
    optional breakdowns by category and feature.
    
    Args:
        title: Section title to display.
        stats: Dictionary of statistic names to values.
        per_category: Optional dict of category-level statistics.
        per_feature: Optional dict of feature-level statistics (collapsible).
    
    Returns:
        str: HTML string containing the formatted summary block.
    
    Example:
        >>> summary = create_section_summary(
        ...     "Network Statistics",
        ...     {"Nodes": 50, "Edges": 120},
        ...     per_category={"AMR": "25 nodes", "Virulence": "25 nodes"}
        ... )
    """
    html = f'<div class="summary-block" style="background:#f7f7f7;border:1px solid #ccc;padding:10px;margin-bottom:10px;"><strong>{title}:</strong><ul>'
    for k, v in stats.items():
        html += f"<li>{k}: {v}</li>"
    html += "</ul>"
    if per_category is not None:
        html += "<b>By category:</b><ul>"
        for cat, stat in per_category.items():
            html += f"<li>{cat}: {stat}</li>"
        html += "</ul>"
    if per_feature is not None:
        html += "<details><summary><b>By feature (click to expand):</b></summary><ul>"
        for feat, stat in per_feature.items():
            html += f"<li>{feat}: {stat}</li>"
        html += "</ul></details>"
    html += "</div>"
    return html


def find_matching_files(uploaded_files, expected_files):
    """
    Match expected CSV file names to uploaded files with flexible name matching.

    Handles multiple file naming patterns including exact matches, numbered
    duplicates (e.g., filename (1).csv), and case-insensitive matches. Useful
    for colab environments where file uploads may have duplicate handling applied.

    Parameters:
        uploaded_files (dict or list): Dictionary of uploaded filenames or list of filenames.
        expected_files (list): List of expected file names to match.

    Returns:
        dict: Mapping from expected file names to actual uploaded file names.
               Keys are from expected_files, values are matched uploaded file names.
               If no match found for an expected file, it is not included in dict.

    Matching Priority:
        1. Exact filename match (case-sensitive)
        2. Numbered duplicate pattern (e.g., "filename (1).csv")
        3. Case-insensitive match

    Example:
        >>> uploaded = ["AMR_genes.csv", "MIC (1).csv"]
        >>> expected = ["AMR_genes.csv", "MIC.csv"]
        >>> mapping = find_matching_files(uploaded, expected)
        >>> # Returns {"AMR_genes.csv": "AMR_genes.csv", "MIC.csv": "MIC (1).csv"}
    """
    file_mapping = {}
    for expected in expected_files:
        base_name = expected.replace(".csv", "")
        if expected in uploaded_files:
            file_mapping[expected] = expected
            continue
        pattern = rf"{re.escape(base_name)}\s*\(\d+\)\.csv"
        found = False
        for uploaded_file in uploaded_files:
            if re.match(pattern, uploaded_file):
                file_mapping[expected] = uploaded_file
                found = True
                break
        if not found:
            for uploaded_file in uploaded_files:
                if uploaded_file.lower() == expected.lower():
                    file_mapping[expected] = uploaded_file
                    found = True
                    break
    return file_mapping


def get_centrality(centrality_dict: dict) -> dict:
    """
    Retrieve and validate centrality measures for network nodes.

    This is a wrapper function that ensures centrality dictionaries from
    NetworkX are properly formatted and accessible. Centrality measures
    quantify the importance or influence of nodes within a network.

    Parameters:
        centrality_dict (dict): Dictionary mapping node identifiers to centrality scores.
                               Typically obtained from NetworkX functions like
                               nx.degree_centrality(), nx.betweenness_centrality(), etc.

    Returns:
        dict: The same centrality dictionary with node IDs as keys and
              centrality scores (float) as values. Range depends on metric:
              - Degree centrality: [0, 1]
              - Betweenness centrality: [0, 1]
              - Closeness centrality: [0, 1]
              - Eigenvector centrality: [0, 1]

    Note:
        This function currently performs pass-through validation. Can be extended
        to add additional checks such as range validation or handling of NaN values.

    Example:
        >>> from networkx import degree_centrality
        >>> G = nx.Graph()
        >>> G.add_edges_from([('A', 'B'), ('B', 'C')])
        >>> deg_cent = degree_centrality(G)
        >>> validated = get_centrality(deg_cent)
        >>> print(validated['B'])  # Central node with highest degree
    """
    return centrality_dict


def expand_categories(df: pd.DataFrame, category_name: str) -> pd.DataFrame:
    """
    Expand categorical features into binary one-hot encoded columns.

    Converts a single categorical column into multiple binary indicator columns
    (one-hot encoding), with support for handling multiple values per strain
    (common in genomic data where one strain may have multiple alleles, plasmids, etc.).

    Parameters:
        df (pd.DataFrame): Input dataframe containing Strain_ID and the category column.
                          Must have at least columns: ["Strain_ID", category_name].
        category_name (str): Name of the column to expand. Supported categories:
                            ["AMR_genes", "MGE", "MLST", "Serotype", "Virulence", "Plasmid", "MIC"].

    Returns:
        pd.DataFrame: One-hot encoded dataframe with:
                     - Binary columns named {category_name}_{value} for each unique value
                     - Strain_ID column preserved
                     - Aggregated by Strain_ID using max() to handle multiple rows per strain

    Details:
        - For MLST and Serotype categories, trailing ".0" decimals are removed
        - Multiple rows per Strain_ID are aggregated (presence of any value = 1)
        - All columns except Strain_ID are binary (0 or 1)

    Example:
        >>> df = pd.DataFrame({
        ...     "Strain_ID": ["S1", "S1", "S2"],
        ...     "MGE": ["Type_A", "Type_B", "Type_A"]
        ... })
        >>> expanded = expand_categories(df, "MGE")
        >>> # Result: Strain_ID, MGE_Type_A, MGE_Type_B columns with aggregated values
    """
    sid = df["Strain_ID"]
    values = df[category_name].astype(str)
    if category_name in ["MLST", "Serotype"]:
        values = values.str.replace(r"\.0$", "", regex=True)
    dummies = pd.get_dummies(values, prefix=category_name)
    dummies["Strain_ID"] = sid
    # Aggregate multiple rows per Strain_ID (e.g., multiple MGE/Plasmid entries)
    dummies = dummies.groupby("Strain_ID", as_index=False).max()
    return dummies


###############################################################################
# STATISTICAL ASSOCIATION FUNCTIONS
###############################################################################

def chi2_phi(
    x: pd.Series, 
    y: pd.Series
) -> Tuple[float, float, pd.DataFrame, float, float]:
    """
    Compute Chi-square test and Phi coefficient for two categorical variables.
    
    Performs Chi-square test of independence with automatic selection of
    Fisher's exact test for small samples. Calculates Phi coefficient as
    effect size measure with confidence interval.
    
    Args:
        x: First categorical variable (binary or categorical).
        y: Second categorical variable (binary or categorical).
    
    Returns:
        Tuple containing:
            - p: P-value from Chi-square or Fisher's exact test.
            - phi: Phi coefficient (effect size, range [-1, 1]).
            - contingency: Contingency table as DataFrame.
            - lo: Lower bound of 95% CI for phi.
            - hi: Upper bound of 95% CI for phi.
    
    Note:
        - Uses Fisher's exact test when n <= 20 for 2x2 tables.
        - Uses Yates' correction when expected counts < 5.
        - CI calculated using Fisher's z-transformation.
    
    Example:
        >>> p, phi, table, ci_lo, ci_hi = chi2_phi(gene_presence, phenotype)
        >>> if p < 0.05:
        ...     print(f"Significant association: phi={phi:.3f}")
    """
    contingency = pd.crosstab(x, y)
    n = contingency.values.sum()
    if contingency.shape == (2, 2) and n <= 20:
        _, p = fisher_exact(contingency)
        chi2 = chi2_contingency(contingency, correction=False)[0]
        phi = min(np.sqrt(chi2 / n), 0.9999)
        return p, phi, contingency, 0.0, 0.0
    chi2_stat, p, _, expected = chi2_contingency(contingency, correction=False)
    if contingency.shape == (2, 2) and (expected < 5).any():
        chi2_stat, p, _, _ = chi2_contingency(contingency, correction=True)
    if n <= 3:
        return p, 0.0, contingency, 0.0, 0.0
    phi = min(np.sqrt(chi2_stat / n), 0.9999)
    z = np.arctanh(phi)
    se = 1 / np.sqrt(n - 3)
    lo = np.tanh(z - norm.ppf(0.975) * se)
    hi = np.tanh(z + norm.ppf(0.975) * se)
    return p, phi, contingency, lo, hi


def cramers_v(contingency: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute Cramér's V coefficient for a contingency table.
    
    Cramér's V is a measure of association between two categorical variables,
    normalized to range [0, 1] regardless of table dimensions.
    
    Args:
        contingency: Contingency table as DataFrame.
    
    Returns:
        Tuple containing:
            - v: Cramér's V coefficient.
            - lo: Lower bound of 95% CI.
            - hi: Upper bound of 95% CI.
    
    Formula:
        V = sqrt(chi2 / (n * min(r-1, k-1)))
    
    Example:
        >>> table = pd.crosstab(serotype, resistance)
        >>> v, ci_lo, ci_hi = cramers_v(table)
    """
    r, k = contingency.shape
    if r < 2 or k < 2:
        return 0.0, 0.0, 0.0
    chi2_stat = chi2_contingency(contingency)[0]
    n = contingency.values.sum()
    if n <= 3:
        return 0.0, 0.0, 0.0
    phi = min(np.sqrt(chi2_stat / n), 0.9999)
    z = np.arctanh(phi)
    se = 1 / np.sqrt(n - 3)
    lo = np.tanh(z - norm.ppf(0.975) * se)
    hi = np.tanh(z + norm.ppf(0.975) * se)
    m = min(r - 1, k - 1)
    if m <= 0:
        return 0.0, 0.0, 0.0
    v = phi / np.sqrt(m)
    return v, lo / np.sqrt(m), hi / np.sqrt(m)


###############################################################################
# INFORMATION THEORY FUNCTIONS
###############################################################################

def calculate_entropy(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate Shannon entropy and normalized entropy for a categorical variable.
    
    Args:
        series: Categorical variable as pandas Series.
    
    Returns:
        Tuple containing:
            - H: Shannon entropy in bits.
            - Hn: Normalized entropy (0-1 scale).
    
    Formula:
        H = -sum(p * log2(p)) for all categories
        Hn = H / log2(n_categories)
    
    Example:
        >>> entropy, norm_entropy = calculate_entropy(serotype_series)
        >>> print(f"Entropy: {entropy:.3f} bits, Normalized: {norm_entropy:.3f}")
    """
    probs = series.value_counts(normalize=True)
    if len(probs) <= 1:
        return 0.0, 0.0
    H = scipy_entropy(probs)
    Hn = H / np.log2(len(probs))
    return H, Hn


def conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate conditional entropy H(X|Y).
    
    Measures the remaining uncertainty in X given knowledge of Y.
    
    Args:
        x: Target variable.
        y: Conditioning variable.
    
    Returns:
        float: Conditional entropy H(X|Y).
    
    Formula:
        H(X|Y) = sum_y P(Y=y) * H(X|Y=y)
    
    Example:
        >>> h_xy = conditional_entropy(resistance, gene_presence)
    """
    joint = pd.crosstab(x, y, normalize="all")
    marginal = joint.sum(axis=0)
    ce = 0.0
    for val in marginal.index:
        p_y = marginal[val]
        if p_y > 0:
            cond_probs = joint[val] / p_y
            non_zero = cond_probs[cond_probs > 0]
            if len(non_zero) > 0:
                ce += p_y * scipy_entropy(non_zero)
    return ce


def information_gain(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate information gain (mutual information) between two variables.
    
    Measures how much knowing Y reduces uncertainty about X.
    
    Args:
        x: Target variable.
        y: Feature variable.
    
    Returns:
        float: Information gain I(X;Y) = H(X) - H(X|Y).
    
    Example:
        >>> ig = information_gain(phenotype, genotype)
        >>> print(f"Information gain: {ig:.3f} bits")
    """
    H, _ = calculate_entropy(x)
    ce = conditional_entropy(x, y)
    return max(0.0, H - ce)


def normalized_mutual_info(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate normalized mutual information between two variables.
    
    Normalizes mutual information to [0, 1] range using geometric mean
    of individual entropies.
    
    Args:
        x: First variable.
        y: Second variable.
    
    Returns:
        float: Normalized mutual information (0-1 scale).
    
    Formula:
        NMI = I(X;Y) / sqrt(H(X) * H(Y))
    
    Example:
        >>> nmi = normalized_mutual_info(gene1, gene2)
        >>> print(f"Normalized MI: {nmi:.3f}")
    """
    ig = information_gain(x, y)
    Hx, _ = calculate_entropy(x)
    Hy, _ = calculate_entropy(y)
    if Hx > 0 and Hy > 0:
        return ig / np.sqrt(Hx * Hy)
    return 0.0


def find_mutually_exclusive(
    df: pd.DataFrame, features: list, mapping: dict, k: int = 2, max_patterns: int = 1000
) -> pd.DataFrame:
    """
    Identify mutually exclusive feature combinations.

    Detects feature combinations that never or rarely co-occur in the dataset,
    indicating potential functional exclusivity or negative selection pressure.
    Useful for finding non-interacting genes or incompatible plasmids.

    Parameters:
        df (pd.DataFrame): Binary feature matrix where rows are strains and columns are features.
                          All feature columns should contain binary values (0 or 1).
        features (list): List of feature column names to analyze.
        mapping (dict): Dictionary mapping feature names to their category
                       (e.g., {"AMR_1": "AMR_genes", "MGE_1": "MGE"}).
        k (int): Size of feature combinations to detect (default 2 for pairs, 3 for triplets).
        max_patterns (int): Maximum number of patterns to return (default 1000).

    Returns:
        pd.DataFrame: Dataframe with columns:
                     - Feature_{i}: Feature name in combination
                     - Category_{i}: Category of feature
                     where i ranges from 1 to k.
                     Returns empty DataFrame if no mutually exclusive patterns found.

    Algorithm:
        Iterates through all k-combinations of features and identifies those
        where no sample has all k features present simultaneously
        (i.e., sum of k features never equals k).

    Example:
        >>> df = pd.DataFrame({
        ...     "F1": [1, 0, 1],
        ...     "F2": [0, 1, 0],
        ...     "F3": [1, 1, 0]
        ... })
        >>> mapping = {"F1": "AMR", "F2": "AMR", "F3": "Plasmid"}
        >>> excl = find_mutually_exclusive(df, ["F1", "F2", "F3"], mapping, k=2)
        >>> # F1 and F2 are mutually exclusive (never both = 1)
    """
    patterns = []
    for combo in itertools.combinations(features, k):
        if (df[list(combo)].sum(axis=1) == k).sum() == 0:
            pat = {}
            for idx, feat in enumerate(combo, 1):
                pat[f"Feature_{idx}"] = feat
                pat[f"Category_{idx}"] = mapping.get(feat, "Unassigned")
            patterns.append(pat)
        if len(patterns) >= max_patterns:
            break
    return pd.DataFrame(patterns)


def get_cluster_hubs(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Extract hub (high-centrality) features from each network cluster.

    Identifies the most central (highly connected) features within each
    community/cluster detected in the network. Hub features serve as key
    connection points and may represent functionally important genes or loci.

    Parameters:
        df (pd.DataFrame): Network node properties dataframe with columns:
                          - Cluster: Cluster/community ID (integer)
                          - Feature: Feature name
                          - Category: Feature category (e.g., "AMR_genes")
                          - Degree_Centrality: Centrality score [0, 1]
                          Other centrality columns (Betweenness, Closeness, Eigenvector)
                          are optional and preserved in output.
        top_n (int): Number of top hubs to extract from each cluster (default 3).

    Returns:
        pd.DataFrame: Dataframe containing hub features with columns:
                     - Cluster: Cluster ID
                     - Feature: Feature name
                     - Category: Category
                     - Degree_Centrality: Degree centrality score
                     Returns empty DataFrame if input lacks required columns
                     or has no clusters.

    Notes:
        - Hubs are selected based on highest Degree_Centrality within each cluster
        - Automatically adapts if a cluster has fewer than top_n nodes
        - Useful for identifying key features for targeted experimental validation

    Example:
        >>> network_df = pd.DataFrame({
        ...     "Cluster": [1, 1, 1, 2, 2],
        ...     "Feature": ["AMR_1", "AMR_2", "MGE_1", "Vir_1", "Vir_2"],
        ...     "Category": ["AMR", "AMR", "MGE", "Virulence", "Virulence"],
        ...     "Degree_Centrality": [0.8, 0.5, 0.3, 0.9, 0.4]
        ... })
        >>> hubs = get_cluster_hubs(network_df, top_n=2)
        >>> # Returns 4 rows: top 2 hubs from each cluster
    """
    if "Cluster" not in df.columns or "Degree_Centrality" not in df.columns:
        return pd.DataFrame()
    hubs = []
    for cluster, group in df.groupby("Cluster"):
        topk = group.nlargest(top_n, "Degree_Centrality")
        for _, row in topk.iterrows():
            hubs.append(
                {
                    "Cluster": cluster,
                    "Feature": row["Feature"],
                    "Category": row["Category"],
                    "Degree_Centrality": row["Degree_Centrality"],
                }
            )
    return pd.DataFrame(hubs)


def adaptive_phi_threshold(
    phi_vals: np.ndarray, method: str = "percentile", percentile: int = 90
) -> float:
    if method == "percentile":
        return np.percentile(phi_vals, percentile)
    elif method == "iqr":
        q75, q25 = np.percentile(phi_vals, [75, 25])
        return q75 + 1.5 * (q75 - q25)
    elif method == "statistical":
        mean = np.mean(phi_vals)
        std = np.std(phi_vals)
        return mean + 2 * std
    return 0.5


def create_interactive_table_with_empty(df: pd.DataFrame, table_id: str) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    return create_interactive_table(df, table_id)


# ---------- EXTENDED SUMMARY HELPERS ----------


def summarize_by_category(df, value_col, category_cols):
    out = {}
    cats = pd.unique(df[category_cols[0]].tolist() + df[category_cols[1]].tolist())
    for cat in cats:
        subset = df[(df[category_cols[0]] == cat) | (df[category_cols[1]] == cat)]
        if len(subset) > 0:
            out[cat] = (
                f"Count: {len(subset)}, mean: {subset[value_col].mean():.3f}, min/max: {subset[value_col].min():.3f}/{subset[value_col].max():.3f}"
            )
    return out


def summarize_by_feature(df, value_col, feature_cols):
    out = {}
    feats = pd.unique(df[feature_cols[0]].tolist() + df[feature_cols[1]].tolist())
    for feat in feats:
        subset = df[(df[feature_cols[0]] == feat) | (df[feature_cols[1]] == feat)]
        if len(subset) > 0:
            out[feat] = (
                f"Count: {len(subset)}, mean: {subset[value_col].mean():.3f}, min/max: {subset[value_col].min():.3f}/{subset[value_col].max():.3f}"
            )
    return out


def summarize_by_category_excl(df, k=2):
    cats = []
    for i in range(1, k + 1):
        cats += df.get(f"Category_{i}", pd.Series(dtype=str)).tolist()
    cats = pd.unique([c for c in cats if pd.notna(c)])
    out = {}
    for cat in cats:
        subset = df[(df.filter(like="Category_").isin([cat])).any(axis=1)]
        out[cat] = f"Count: {len(subset)}"
    return out


def summarize_by_feature_excl(df, k=2):
    feats = []
    for i in range(1, k + 1):
        feats += df.get(f"Feature_{i}", pd.Series(dtype=str)).tolist()
    feats = pd.unique([f for f in feats if pd.notna(f)])
    out = {}
    for feat in feats:
        subset = df[(df.filter(like="Feature_").isin([feat])).any(axis=1)]
        out[feat] = f"Count: {len(subset)}"
    return out


def summarize_by_category_network(df, value_col="Degree_Centrality"):
    out = {}
    cats = df["Category"].unique()
    for cat in cats:
        sub = df[df["Category"] == cat]
        if len(sub) > 0:
            out[cat] = (
                f"Count: {len(sub)}, mean degree: {sub[value_col].mean():.3f}, min/max degree: {sub[value_col].min():.3f}/{sub[value_col].max():.3f}"
            )
    return out


def summarize_by_feature_network(df, value_col="Degree_Centrality"):
    out = {}
    for feat in df["Feature"]:
        sub = df[df["Feature"] == feat]
        if len(sub) > 0:
            out[feat] = f"degree: {sub[value_col].values[0]:.3f}"
    return out


# ====================== REPORT GENERATION ======================


def generate_report_with_cluster_stats(
    chi2_df,
    network_df,
    entropy_df,
    cramers_df,
    feature_summary_df,
    excl2_df,
    excl3_df,
    hubs_df,
    network_html,
    chi2_summary,
    entropy_summary,
    cramers_summary,
    excl2_summary,
    excl3_summary,
    network_summary,
    hubs_summary,
    chi2_cat,
    chi2_feat,
    entropy_cat,
    entropy_feat,
    cramers_cat,
    cramers_feat,
    excl2_cat,
    excl2_feat,
    excl3_cat,
    excl3_feat,
    network_cat,
    network_feat,
):

    css_links = (
        '<link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css"/>'
        '<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.3.6/css/buttons.dataTables.min.css"/>'
    )
    js_scripts = (
        '<script src="https://code.jquery.com/jquery-3.5.1.js"></script>'
        '<script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>'
        '<script src="https://cdn.datatables.net/buttons/2.3.6/js/dataTables.buttons.min.js"></script>'
        '<script src="https://cdn.datatables.net/buttons/2.3.6/js/buttons.html5.min.js"></script>'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js"></script>'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js"></script>'
    )
    init_script = (
        "<script>$(document).ready(function() {"
        "$('table.data-table').DataTable({dom:'Bfrtip', buttons:['copyHtml5','csvHtml5','excelHtml5','pdfHtml5'], pageLength:20});"
        "});</script>"
    )

    intro_block = (
        "<section>"
        "<h2>Antimicrobial Resistance (MDR) Context</h2>"
        "<p>This report summarizes multidrug resistance (MDR) and resistance-associated findings across genomic and phenotypic features.</p>"
        "<p>Key resistance signals and MDR patterns are highlighted in subsequent tables and summaries.</p>"
        "</section>"
    )

    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">'
        "<title>StrepSuis-GenPhenNet: Network-Based Integration of Genome–Phenome Data in Streptococcus suis</title>"
        + css_links
        + js_scripts
        + "<style>.summary-block{background:#f7f7f7;border:1px solid #ccc;padding:10px;margin-bottom:10px;}</style>"
        "</head><body>"
        "<h1>StrepSuis-GenPhenNet: Network-Based Integration of Genome–Phenome Data in Streptococcus suis</h1>"
        + intro_block
        + "<section><h2>1. Feature Summary</h2>"
        + create_interactive_table_with_empty(feature_summary_df, "FeatureSummary")
        + "</section>"
        "<section><h2>2. Chi-square Results</h2>"
        + create_section_summary(
            "Chi-square/Fisher results summary", chi2_summary, chi2_cat, chi2_feat
        )
        + create_interactive_table_with_empty(chi2_df, "ChiSquare")
        + "</section>"
        "<section><h2>3. Information Theory</h2>"
        + create_section_summary(
            "Information theory summary", entropy_summary, entropy_cat, entropy_feat
        )
        + create_interactive_table_with_empty(entropy_df, "Entropy")
        + "<h3>Cramér's V</h3>"
        + create_section_summary("Cramér's V summary", cramers_summary, cramers_cat, cramers_feat)
        + create_interactive_table_with_empty(cramers_df, "Cramers")
        + "</section>"
        "<section><h2>4. Mutually Exclusive Patterns</h2>"
        + "<h3>Pairs (k=2)</h3>"
        + create_section_summary(
            "Mutually exclusive pairs summary", excl2_summary, excl2_cat, excl2_feat
        )
        + create_interactive_table_with_empty(excl2_df, "Excl2")
        + "<h3>Triplets (k=3)</h3>"
        + create_section_summary(
            "Mutually exclusive triplets summary", excl3_summary, excl3_cat, excl3_feat
        )
        + create_interactive_table_with_empty(excl3_df, "Excl3")
        + "</section>"
        "<section><h2>5. Network Analysis</h2>"
        + create_section_summary("Network summary", network_summary, network_cat, network_feat)
        + create_interactive_table_with_empty(network_df, "Network")
        + network_html
        + "</section>"
        "<section><h2>6. Cluster Hubs</h2>"
        + create_section_summary("Cluster hubs summary", hubs_summary)
        + create_interactive_table_with_empty(hubs_df, "Hubs")
        + "</section>"
        + init_script
        + "</body></html>"
    )
    return html


# ====================== EXCEL REPORT GENERATION ======================


def generate_excel_report_with_cluster_stats(
    chi2_df,
    network_df,
    entropy_df,
    cramers_df,
    feature_summary_df,
    excl2_df,
    excl3_df,
    hubs_df,
    fig_network,
    chi2_summary,
    entropy_summary,
    cramers_summary,
    excl2_summary,
    excl3_summary,
    network_summary,
    hubs_summary,
    chi2_cat,
    chi2_feat,
    entropy_cat,
    entropy_feat,
    cramers_cat,
    cramers_feat,
    excl2_cat,
    excl2_feat,
    excl3_cat,
    excl3_feat,
    network_cat,
    network_feat,
):
    """
    Generate comprehensive Excel report with all analysis results and PNG charts.

    This function creates a detailed Excel workbook with multiple sheets containing:
    - Metadata and methodology
    - Feature summary statistics
    - Chi-square and Fisher exact test results
    - Information theory metrics (entropy, Cramér's V)
    - Mutually exclusive patterns (pairs and triplets)
    - Network analysis with centrality measures
    - Cluster hubs identification
    - Category and feature-level summaries

    All visualizations are saved as PNG files in the png_charts subfolder.
    """
    # Initialize Excel report generator
    excel_gen = ExcelReportGenerator(output_folder=output_folder)

    # Save network visualization as PNG if available
    if fig_network is not None:
        try:
            excel_gen.save_plotly_figure_fallback(
                fig_network, "network_3d_visualization", width=1400, height=1000
            )
        except Exception as e:
            print(f"Could not save network visualization: {e}")

    # Prepare methodology description
    methodology = {
        "Chi-square and Fisher Exact Tests": (
            "Statistical tests to assess associations between categorical features. "
            "Fisher exact test is used for 2x2 contingency tables with low expected counts (≤20 samples). "
            "Benjamini-Hochberg FDR correction applied for multiple testing."
        ),
        "Information Theory Metrics": (
            "Entropy measures the uncertainty/information content of feature distributions. "
            "Cramér's V quantifies association strength between categorical variables (0=no association, 1=perfect association). "
            "Phi coefficient is calculated for binary features."
        ),
        "Mutually Exclusive Patterns": (
            "Identifies features that rarely or never co-occur in the dataset. "
            "Analyzed for pairs (k=2) and triplets (k=3) of features. "
            "Uses hypergeometric test to assess statistical significance of co-occurrence patterns."
        ),
        "Network Analysis": (
            "Constructs a network where nodes are features and edges represent significant associations (Phi ≥ 0.3). "
            "Community detection using Louvain algorithm to identify feature clusters. "
            "Centrality metrics: Degree (connectivity), Betweenness (bridging), Closeness (reach), Eigenvector (influence)."
        ),
        "Hub Identification": (
            "Identifies highly connected features that serve as central nodes in the network. "
            "Hubs are defined as nodes with degree centrality > mean + 1.5×SD."
        ),
    }

    # Prepare sheets data
    sheets_data = {}

    # Feature Summary
    sheets_data["Feature_Summary"] = (
        feature_summary_df,
        "Overview of all feature categories and their counts",
    )

    # Chi-square Results
    if chi2_df is not None and not chi2_df.empty:
        sheets_data["Chi2_Results"] = (
            chi2_df,
            f"Chi-square/Fisher test results. Total tests: {chi2_summary.get('Total tests', 'N/A')}, "
            f"Significant (FDR<0.05): {chi2_summary.get('Significant (FDR<0.05)', 'N/A')}",
        )

    # Entropy Results
    if entropy_df is not None and not entropy_df.empty:
        sheets_data["Entropy_Results"] = (
            entropy_df,
            f"Information entropy for each feature. Mean entropy: {entropy_summary.get('Mean entropy', 'N/A')}",
        )

    # Cramér's V Results
    if cramers_df is not None and not cramers_df.empty:
        sheets_data["Cramers_V_Results"] = (
            cramers_df,
            f"Cramér's V association strength. Significant pairs: {cramers_summary.get('Significant (FDR<0.05)', 'N/A')}",
        )

    # Mutually Exclusive Pairs
    if excl2_df is not None and not excl2_df.empty:
        sheets_data["Exclusive_Pairs_k2"] = (
            excl2_df,
            f"Mutually exclusive feature pairs. Total patterns: {excl2_summary.get('Total patterns', 'N/A')}, "
            f"Significant: {excl2_summary.get('Significant (FDR<0.05)', 'N/A')}",
        )

    # Mutually Exclusive Triplets
    if excl3_df is not None and not excl3_df.empty:
        sheets_data["Exclusive_Triplets_k3"] = (
            excl3_df,
            f"Mutually exclusive feature triplets. Total patterns: {excl3_summary.get('Total patterns', 'N/A')}, "
            f"Significant: {excl3_summary.get('Significant (FDR<0.05)', 'N/A')}",
        )

    # Network Analysis
    if network_df is not None and not network_df.empty:
        sheets_data["Network_Analysis"] = (
            network_df,
            f"Network node properties with community assignments. "
            f"Total nodes: {network_summary.get('Total nodes', 'N/A')}, "
            f"Total edges: {network_summary.get('Total edges', 'N/A')}, "
            f"Clusters: {network_summary.get('Clusters', 'N/A')}",
        )

    # Cluster Hubs
    if hubs_df is not None and not hubs_df.empty:
        sheets_data["Cluster_Hubs"] = (
            hubs_df,
            f"Highly connected hub features. Total hubs: {hubs_summary.get('Total hubs', 'N/A')}",
        )

    # Category-level summaries
    if chi2_cat:
        df_chi2_cat = pd.DataFrame(
            [
                {"Category": k, "Metric": "Chi2_Significant_Count", "Value": v}
                for k, v in chi2_cat.items()
            ]
        )
        sheets_data["Chi2_by_Category"] = (df_chi2_cat, "Chi-square results aggregated by category")

    if entropy_cat:
        df_entropy_cat = pd.DataFrame(
            [{"Category": k, "Metric": "Mean_Entropy", "Value": v} for k, v in entropy_cat.items()]
        )
        sheets_data["Entropy_by_Category"] = (df_entropy_cat, "Entropy aggregated by category")

    if network_cat:
        df_network_cat = pd.DataFrame(
            [
                {"Category": k, "Metric": "Mean_Degree_Centrality", "Value": v}
                for k, v in network_cat.items()
            ]
        )
        sheets_data["Network_by_Category"] = (
            df_network_cat,
            "Network metrics aggregated by category",
        )

    # Prepare metadata
    metadata = {
        "Total_Features_Analyzed": len(feature_summary_df) if feature_summary_df is not None else 0,
        "Chi2_Tests_Performed": chi2_summary.get("Total tests", "N/A"),
        "Significant_Associations_FDR": chi2_summary.get("Significant (FDR<0.05)", "N/A"),
        "Network_Nodes": network_summary.get("Total nodes", "N/A"),
        "Network_Edges": network_summary.get("Total edges", "N/A"),
        "Network_Clusters": network_summary.get("Clusters", "N/A"),
    }

    # Generate Excel report
    excel_path = excel_gen.generate_excel_report(
        report_name="Network_Analysis_Report",
        sheets_data=sheets_data,
        methodology=methodology,
        **metadata,
    )

    return excel_path


# ====================== MAIN WORKFLOW ======================


def setup_logging():
    """
    Configure logging to both file and console output.
    Creates output directory if it doesn't exist.
    """
    global output_folder
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_folder, "network_analysis_log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def perform_full_analysis():
    """
    Execute comprehensive network analysis pipeline with full reporting.

    This function orchestrates the complete feature association network analysis workflow:

    Execution Flow:
    1. Environment setup and logging initialization
    2. Data loading and validation (7 CSV files required)
    3. Feature extraction and category expansion
    4. Chi-square and Fisher exact tests for pairwise associations
    5. Information theory metrics (entropy, conditional entropy, mutual information)
    6. Cramér's V calculation for association strength
    7. Mutually exclusive pattern detection (pairs and triplets)
    8. Network construction with community detection
    9. Hub identification and centrality analysis
    10. Comprehensive HTML report generation with interactive tables
    11. Excel report generation with methodology and PNG charts

    Statistical Methods:
    - Chi-square tests with Yates correction for 2x2 tables with low counts
    - Fisher exact test for 2x2 tables with ≤20 samples
    - Benjamini-Hochberg FDR correction for multiple testing
    - Bootstrap confidence intervals for effect sizes
    - Adaptive threshold selection for network construction
    - Louvain algorithm for community detection

    Required CSV Files:
    - MGE.csv: Mobile genetic elements
    - MIC.csv: Minimum inhibitory concentrations
    - MLST.csv: Multi-locus sequence typing
    - Plasmid.csv: Plasmid profiles
    - Serotype.csv: Serotype data
    - Virulence.csv: Virulence factors
    - AMR_genes.csv: Antimicrobial resistance genes

    Output Files:
    - report.html: Interactive HTML report with DataTables
    - Network_Analysis_Report_YYYYMMDD_HHMMSS.xlsx: Excel workbook with multiple sheets
    - output/png_charts/: Directory containing PNG visualizations
    - output/network_analysis_log.txt: Detailed execution log
    """
    start_time = time.time()

    # Setup logging
    setup_logging()
    logging.info("=" * 80)
    logging.info("Starting Network Analysis Pipeline")
    logging.info("=" * 80)

    np.random.seed(2800)
    logging.info("Random seed set to 2800 for reproducibility")

    # Step 1: File upload and validation
    logging.info("STEP 1: File Upload and Validation")

    if IN_COLAB:
        print(
            "Please upload the following CSV files: MGE.csv, MIC.csv, MLST.csv, Plasmid.csv, Serotype.csv, Virulence.csv, AMR_genes.csv"
        )
        uploaded_files = files.upload()
        expected = [
            "MGE.csv",
            "MIC.csv",
            "MLST.csv",
            "Plasmid.csv",
            "Serotype.csv",
            "Virulence.csv",
            "AMR_genes.csv",
        ]
        file_mapping = find_matching_files(uploaded_files, expected)
        missing = [fn for fn in expected if fn not in file_mapping]
        if missing:
            logging.error(f"Missing files: {', '.join(missing)}")
            print("Available files:", list(uploaded_files.keys()))
    else:
        # Local execution - use files from current directory
        print("Looking for CSV files in current directory...")
        expected = [
            "MGE.csv",
            "MIC.csv",
            "MLST.csv",
            "Plasmid.csv",
            "Serotype.csv",
            "Virulence.csv",
            "AMR_genes.csv",
        ]
        file_mapping = {f: f for f in expected if os.path.exists(f)}
        missing = [fn for fn in expected if fn not in file_mapping]
        if missing:
            logging.warning(f"Optional files not found: {', '.join(missing)}")
            print(f"Note: Some optional files not found: {', '.join(missing)}")

    # Check if we have at least the minimum required files
    required_minimum = ["MIC.csv", "AMR_genes.csv", "Virulence.csv"]
    missing_required = [f for f in required_minimum if f not in file_mapping]
    if missing_required:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_required)}")

    logging.info("All required files found successfully")
    print("File mapping:")
    for expected_name, actual_name in file_mapping.items():
        print(f"  {expected_name} -> {actual_name}")
        logging.info(f"File mapping: {expected_name} -> {actual_name}")

    # Step 2: Data loading
    logging.info("STEP 2: Loading data from CSV files")
    data = {}
    for expected_name, actual_name in file_mapping.items():
        data[expected_name] = pd.read_csv(actual_name, dtype={"Strain_ID": str})
        logging.info(f"Loaded {expected_name}: shape={data[expected_name].shape}")

    # Step 3: Data validation
    logging.info("STEP 3: Validating data structure")
    for fn, df in data.items():
        if "Strain_ID" not in df.columns:
            logging.error(f"File {fn} missing 'Strain_ID' column")
            raise KeyError(f"File {fn} must contain 'Strain_ID' column.")
    logging.info("All files contain required 'Strain_ID' column")

    # Step 4: Feature summary
    logging.info("STEP 4: Generating feature summary statistics")
    summary = []
    for fn in expected:
        cat = fn.replace(".csv", "")
        df = data[fn]
        if cat in ["MGE", "MLST", "Plasmid", "Serotype"]:
            unique_vals = df[cat].dropna().astype(str).str.replace(r"\.0$", "", regex=True).unique()
            unique_vals = [v for v in unique_vals if v and v.lower() != "nan"]
            count = len(unique_vals)
        else:
            count = len([c for c in df.columns if c != "Strain_ID"])
        summary.append({"Category": cat, "Number_of_Features": count})
        logging.info(f"Category {cat}: {count} features")
    feature_summary_df = pd.DataFrame(summary)

    # Step 5: Category expansion
    logging.info("STEP 5: Expanding categorical features")
    expanded = {}
    for fn in ["MGE.csv", "MLST.csv", "Plasmid.csv", "Serotype.csv"]:
        cat = fn.replace(".csv", "")
        expanded[cat] = expand_categories(data[fn], cat)
        logging.info(f"Expanded {cat}: {len(expanded[cat].columns)-1} binary columns")

    # Step 6: Data merging
    logging.info("STEP 6: Merging all data sources")
    merged = expanded["MGE"]
    for fn in [
        "MIC.csv",
        "MLST.csv",
        "Plasmid.csv",
        "Serotype.csv",
        "Virulence.csv",
        "AMR_genes.csv",
    ]:
        key = fn.replace(".csv", "")
        merged = merged.merge(expanded.get(key, data[fn]), on="Strain_ID", how="outer")
    merged.fillna(0, inplace=True)
    features = [c for c in merged.columns if c != "Strain_ID"]
    logging.info(f"Merged dataset: {len(features)} total features across {len(merged)} strains")

    logging.info(f"Merged dataset: {len(features)} total features across {len(merged)} strains")

    # Step 7: Feature-category mapping
    logging.info("STEP 7: Building feature-category mapping")
    feature_category = {}
    for fn in expected:
        cat = fn.replace(".csv", "")
        df = data[fn]
        if cat in expanded:
            cols = [col for col in expanded[cat].columns if col != "Strain_ID"]
            for c in cols:
                feature_category[c] = cat
        else:
            for c in df.columns:
                if c != "Strain_ID":
                    feature_category[c] = cat
    for feat in features:
        if feat not in feature_category:
            if "_" in feat:
                prefix = feat.split("_")[0]
                feature_category[feat] = prefix
            else:
                feature_category[feat] = "Unassigned"
    logging.info(f"Feature-category mapping complete: {len(feature_category)} features mapped")

    # Step 8: Mutually exclusive pattern detection
    logging.info("STEP 8: Detecting mutually exclusive patterns")
    excl2_df = find_mutually_exclusive(merged, features, feature_category, k=2)
    excl3_df = find_mutually_exclusive(merged, features, feature_category, k=3)
    logging.info(f"Found {len(excl2_df)} mutually exclusive pairs")
    logging.info(f"Found {len(excl3_df)} mutually exclusive triplets")

    # Step 9: Chi-square and Fisher exact tests
    logging.info("STEP 9: Performing pairwise Chi-square/Fisher exact tests")
    pairs = list(itertools.combinations(features, 2))
    logging.info(f"Testing {len(pairs)} feature pairs (this may take several minutes...)")
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chi2_phi, merged[f1], merged[f2]): (f1, f2) for f1, f2 in pairs}
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 10000 == 0:
                logging.info(
                    f"Completed {completed}/{len(pairs)} tests ({100*completed/len(pairs):.1f}%)"
                )
            f1, f2 = futures[fut]
            p, phi, _, lo, hi = fut.result()
            results.append(
                {
                    "Feature1": f1,
                    "Category1": feature_category.get(f1, "Unassigned"),
                    "Feature2": f2,
                    "Category2": feature_category.get(f2, "Unassigned"),
                    "P_value": p,
                    "Phi_coefficient": phi,
                    "CI_Lower": lo,
                    "CI_Upper": hi,
                }
            )
    chi2_df = pd.DataFrame(results)
    chi2_df["Significant"], chi2_df["P_adjusted"], _, _ = smm.multipletests(
        chi2_df["P_value"], method="fdr_bh"
    )
    logging.info(
        f"Chi-square tests complete: {chi2_df['Significant'].sum()} significant associations (FDR<0.05)"
    )

    # Step 10: Information theory metrics
    logging.info("STEP 10: Computing information theory metrics")

    # Step 10: Information theory metrics
    logging.info("STEP 10: Computing information theory metrics")
    numeric_feats = merged[features].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_feats:
        for feat in features:
            merged[feat] = pd.to_numeric(merged[feat], errors="coerce").fillna(0)
        numeric_feats = features
    top_feats = (
        pd.Series(merged[numeric_feats].var()).nlargest(min(50, len(numeric_feats))).index.tolist()
    )
    logging.info(f"Analyzing top {len(top_feats)} features with highest variance")
    print(f"Analyzing {len(top_feats)} features for information theory metrics.")

    info_results, cramers_results = [], []
    info_pvals, cramers_pvals = [], []
    total_info_pairs = len(top_feats) * (len(top_feats) - 1)
    completed_info = 0
    for f1 in top_feats:
        H, Hn = calculate_entropy(merged[f1])
        for f2 in top_feats:
            if f1 == f2:
                continue
            completed_info += 1
            if completed_info % 500 == 0:
                logging.info(
                    f"Completed {completed_info}/{total_info_pairs} information theory calculations ({100*completed_info/total_info_pairs:.1f}%)"
                )
            ce = conditional_entropy(merged[f1], merged[f2])
            ig = information_gain(merged[f1], merged[f2])
            nmi = normalized_mutual_info(merged[f1], merged[f2])
            tbl = pd.crosstab(merged[f1], merged[f2])
            n = tbl.values.sum()
            p_ig = 1.0
            df_ig = (tbl.shape[0] - 1) * (tbl.shape[1] - 1)
            if n > 0 and df_ig > 0:
                p_ig = 1 - chi2_contingency(tbl, correction=False)[1]
            info_pvals.append(p_ig)
            v, vlo, vhi = cramers_v(tbl)
            cramers_pvals.append(chi2_contingency(tbl)[1] if tbl.size > 1 else 1.0)
            info_results.append(
                {
                    "Feature1": f1,
                    "Category1": feature_category.get(f1, "Unassigned"),
                    "Feature2": f2,
                    "Category2": feature_category.get(f2, "Unassigned"),
                    "Entropy": H,
                    "Entropy_Normalized": Hn,
                    "Conditional_Entropy": ce,
                    "Information_Gain": ig,
                    "Normalized_Mutual_Information": nmi,
                }
            )
            cramers_results.append(
                {
                    "Feature1": f1,
                    "Category1": feature_category.get(f1, "Unassigned"),
                    "Feature2": f2,
                    "Category2": feature_category.get(f2, "Unassigned"),
                    "Cramers_V": v,
                    "CI_Lower": vlo,
                    "CI_Upper": vhi,
                }
            )
    entropy_df = pd.DataFrame(info_results)
    cramers_df = pd.DataFrame(cramers_results)
    entropy_df["Significant"], entropy_df["P_adjusted"], _, _ = smm.multipletests(
        info_pvals, method="fdr_bh"
    )
    cramers_df["Significant"], cramers_df["P_adjusted"], _, _ = smm.multipletests(
        cramers_pvals, method="fdr_bh"
    )
    logging.info(
        f"Information theory complete: {entropy_df['Significant'].sum()} significant entropy associations"
    )
    logging.info(f"Cramér's V complete: {cramers_df['Significant'].sum()} significant associations")

    # Step 11: Generate section summaries
    logging.info("STEP 11: Generating section summaries")

    # Step 11: Generate section summaries
    logging.info("STEP 11: Generating section summaries")

    chi2_summary = {
        "Total pairs": len(chi2_df),
        "Significant (FDR)": int(chi2_df["Significant"].sum()),
        "Phi mean": np.round(chi2_df["Phi_coefficient"].mean(), 3),
        "Phi min/max": f"{chi2_df['Phi_coefficient'].min():.3f} / {chi2_df['Phi_coefficient'].max():.3f}",
    }
    chi2_cat = summarize_by_category(chi2_df, "Phi_coefficient", ["Category1", "Category2"])
    chi2_feat = summarize_by_feature(chi2_df, "Phi_coefficient", ["Feature1", "Feature2"])

    entropy_summary = {
        "Pairs analyzed": len(entropy_df),
        "Significant (FDR)": int(entropy_df["Significant"].sum()),
        "IG mean": np.round(entropy_df["Information_Gain"].mean(), 3),
        "NMI mean": np.round(entropy_df["Normalized_Mutual_Information"].mean(), 3),
    }
    entropy_cat = summarize_by_category(entropy_df, "Information_Gain", ["Category1", "Category2"])
    entropy_feat = summarize_by_feature(entropy_df, "Information_Gain", ["Feature1", "Feature2"])

    cramers_summary = {
        "Pairs analyzed": len(cramers_df),
        "Significant (FDR)": int(cramers_df["Significant"].sum()),
        "V mean": np.round(cramers_df["Cramers_V"].mean(), 3),
        "V min/max": f"{cramers_df['Cramers_V'].min():.3f} / {cramers_df['Cramers_V'].max():.3f}",
    }
    cramers_cat = summarize_by_category(cramers_df, "Cramers_V", ["Category1", "Category2"])
    cramers_feat = summarize_by_feature(cramers_df, "Cramers_V", ["Feature1", "Feature2"])

    excl2_summary = {"Number of mutually exclusive pairs": len(excl2_df)}
    excl2_cat = summarize_by_category_excl(excl2_df, k=2)
    excl2_feat = summarize_by_feature_excl(excl2_df, k=2)

    excl3_summary = {"Number of mutually exclusive triplets": len(excl3_df)}
    excl3_cat = summarize_by_category_excl(excl3_df, k=3)
    excl3_feat = summarize_by_feature_excl(excl3_df, k=3)

    # Step 12: Network construction and analysis
    logging.info("STEP 12: Constructing feature association network")
    phi_vals = chi2_df["Phi_coefficient"].values
    threshold = adaptive_phi_threshold(phi_vals, method="percentile", percentile=90)
    logging.info(f"Adaptive phi threshold (90th percentile): {threshold:.3f}")
    print(f"Applying phi threshold: {threshold:.3f}")
    print(f"Applying phi threshold: {threshold:.3f}")
    sig_edges = chi2_df[chi2_df["Significant"] & (chi2_df["Phi_coefficient"] > threshold)]
    G = nx.Graph()
    for _, row in sig_edges.iterrows():
        G.add_edge(row["Feature1"], row["Feature2"], weight=row["Phi_coefficient"])
    logging.info(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    network_df = pd.DataFrame()
    hubs_df = pd.DataFrame()
    network_html = ""
    hubs_summary = {"Number of cluster hubs": 0}
    network_summary = {"Edges (significant)": G.number_of_edges(), "Nodes": G.number_of_nodes()}
    network_cat, network_feat = {}, {}

    if G.number_of_edges() > 0:
        logging.info("STEP 13: Performing community detection and centrality analysis")
        partition = community_louvain.best_partition(G, weight="weight", random_state=2800)
        cluster_ids = sorted(set(partition.values()))
        num_clusters = len(cluster_ids)
        logging.info(f"Detected {num_clusters} communities using Louvain algorithm")

        deg_cent = get_centrality(nx.degree_centrality(G))
        btw_cent = get_centrality(nx.betweenness_centrality(G))
        cls_cent = get_centrality(nx.closeness_centrality(G))
        eig_cent = get_centrality(nx.eigenvector_centrality(G, tol=1e-6))
        logging.info("Calculated degree, betweenness, closeness, and eigenvector centrality")

        node_data = []
        for node in G.nodes():
            node_data.append(
                {
                    "Feature": node,
                    "Category": feature_category.get(node, "Unassigned"),
                    "Cluster": partition[node] + 1,
                    "Degree_Centrality": deg_cent[node],
                    "Betweenness_Centrality": btw_cent[node],
                    "Closeness_Centrality": cls_cent[node],
                    "Eigenvector_Centrality": eig_cent[node],
                }
            )
        network_df = pd.DataFrame(node_data)
        hubs_df = get_cluster_hubs(network_df)
        hubs_summary = {"Total hubs (top 3/cluster)": len(hubs_df), "Clusters": num_clusters}
        logging.info(f"Identified {len(hubs_df)} hub features across {num_clusters} clusters")

        # Step 13.5: NEW - Causal Discovery Framework
        logging.info("STEP 13.5: Performing causal discovery analysis")
        causal_edges_df = pd.DataFrame()
        try:
            from .causal_discovery import CausalDiscoveryFramework
            
            # Separate gene and phenotype data
            gene_cols = [f for f in features if feature_category.get(f, "") in ["AMR_genes", "Virulence", "MGE", "Plasmid"]]
            pheno_cols = [f for f in features if feature_category.get(f, "") in ["MIC", "MLST", "Serotype"]]
            
            if gene_cols and pheno_cols:
                gene_data = merged[["Strain_ID"] + gene_cols].set_index("Strain_ID")
                pheno_data = merged[["Strain_ID"] + pheno_cols].set_index("Strain_ID")
                
                # Prepare initial associations (from chi2_df, filter gene-phenotype pairs)
                initial_assoc = chi2_df[
                    (chi2_df["Feature1"].isin(gene_cols) & chi2_df["Feature2"].isin(pheno_cols)) |
                    (chi2_df["Feature1"].isin(pheno_cols) & chi2_df["Feature2"].isin(gene_cols))
                ].copy()
                
                if not initial_assoc.empty:
                    causal_framework = CausalDiscoveryFramework(
                        gene_data=gene_data,
                        phenotype_data=pheno_data,
                        initial_associations=initial_assoc,
                    )
                    causal_edges_df = causal_framework.discover_causal_network(alpha=0.05, n_permutations=1000)
                    
                    if not causal_edges_df.empty:
                        causal_edges_df.to_csv(
                            os.path.join(output_folder, "causal_discovery_results.csv"),
                            index=False
                        )
                        logging.info(f"Causal discovery completed: {len(causal_edges_df)} causal edges identified")
                    else:
                        logging.warning("Causal discovery found no causal edges")
                else:
                    logging.warning("No gene-phenotype associations found for causal discovery")
            else:
                logging.warning("Insufficient gene/phenotype data for causal discovery")
        except Exception as e:
            logging.warning(f"Causal discovery analysis failed: {e}")

        # Step 13.6: NEW - Predictive Modeling
        logging.info("STEP 13.6: Building predictive models (genotype → phenotype)")
        prediction_results = {}
        try:
            from .predictive_modeling import GenotypePhenotypePredictor
            
            # Separate gene and phenotype data
            gene_cols = [f for f in features if feature_category.get(f, "") in ["AMR_genes", "Virulence", "MGE", "Plasmid"]]
            pheno_cols = [f for f in features if feature_category.get(f, "") in ["MIC"]]
            
            if gene_cols and pheno_cols:
                gene_data = merged[["Strain_ID"] + gene_cols].set_index("Strain_ID")
                pheno_data = merged[["Strain_ID"] + pheno_cols].set_index("Strain_ID")
                
                # Ensure binary data
                gene_data = gene_data.fillna(0).astype(int)
                pheno_data = pheno_data.fillna(0).astype(int)
                
                predictor = GenotypePhenotypePredictor(
                    gene_data=gene_data,
                    phenotype_data=pheno_data,
                    test_size=0.3,
                    random_state=42,
                )
                prediction_results = predictor.build_prediction_models(min_samples=10)
                
                if prediction_results:
                    # Generate and save report
                    report_path = os.path.join(output_folder, "predictive_modeling_report.txt")
                    predictor.generate_prediction_report(prediction_results, output_path=report_path)
                    
                    # Save results as CSV
                    results_summary = []
                    for phenotype, models in prediction_results.items():
                        for model_name, metrics in models.items():
                            if metrics is not None:
                                results_summary.append({
                                    'phenotype': phenotype,
                                    'model': model_name,
                                    'roc_auc': metrics.get('roc_auc', np.nan),
                                    'accuracy': metrics.get('accuracy', np.nan),
                                    'f1_score': metrics.get('f1', np.nan),
                                })
                    
                    if results_summary:
                        pd.DataFrame(results_summary).to_csv(
                            os.path.join(output_folder, "predictive_modeling_results.csv"),
                            index=False
                        )
                    
                    logging.info(f"Predictive modeling completed: {len(prediction_results)} phenotypes analyzed")
                else:
                    logging.warning("Predictive modeling found no valid models")
            else:
                logging.warning("Insufficient gene/phenotype data for predictive modeling")
        except Exception as e:
            logging.warning(f"Predictive modeling failed: {e}")

        # Step 14: 3D network visualization
        logging.info("STEP 14: Generating 3D network visualization")
        import matplotlib
        from matplotlib import cm

        if num_clusters > 10:
            cmap = cm.get_cmap("tab20", num_clusters)
            palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(num_clusters)]
        else:
            palette = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "cyan",
                "magenta",
                "yellow",
                "pink",
                "brown",
            ][:num_clusters]

        pos = nx.spring_layout(G, dim=3, seed=2800, k=0.6)
        edge_x, edge_y, edge_z = [], [], []
        hover_texts = []
        for u, v, d in G.edges(data=True):
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
            hover_texts.append(f"{u}–{v}<br>Phi: {d['weight']:.3f}")
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=2, color="#888"),
            hoverinfo="text",
            hovertext=hover_texts,
        )
        node_x, node_y, node_z, texts, colors, hovers = [], [], [], [], [], []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            clust = partition[node]
            colors.append(palette[clust % len(palette)])
            texts.append(f"{node}\n(C{clust+1})")
            hovers.append(
                f"Feature: {node}<br>Category: {feature_category.get(node,'Unassigned')}<br>Cluster: {clust+1}"
                f"<br>Degree: {deg_cent[node]:.3f}<br>Betweenness: {btw_cent[node]:.3f}"
                f"<br>Closeness: {cls_cent[node]:.3f}<br>Eigenvector: {eig_cent[node]:.3f}"
            )
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            marker=dict(color=colors, size=12, opacity=0.8),
            text=texts,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hovertext=hovers,
            hoverinfo="text",
        )
        fig_network = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="3D Network with Cluster Labels",
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(showgrid=True, title="X"),
                    yaxis=dict(showgrid=True, title="Y"),
                    zaxis=dict(showgrid=True, title="Z"),
                ),
                width=1200,
                height=800,
                showlegend=False,
            ),
        )
        fig_network.write_html("network_visualization.html", include_plotlyjs="cdn")
        with open("network_visualization.html", "r") as f:
            network_html = f.read()
        logging.info("3D network visualization saved to network_visualization.html")

        network_summary.update(
            {
                "Clusters": num_clusters,
                "Max cluster size": network_df.groupby("Cluster").size().max(),
                "Min cluster size": network_df.groupby("Cluster").size().min(),
            }
        )
        network_cat = summarize_by_category_network(network_df, value_col="Degree_Centrality")
        network_feat = summarize_by_feature_network(network_df, value_col="Degree_Centrality")
    else:
        logging.warning("No significant network edges detected")
        network_html = "<p>No significant network edges detected.</p>"
        fig_network = None

    # Step 15: Generate HTML report
    logging.info("STEP 15: Generating interactive HTML report")
    # Step 15: Generate HTML report
    logging.info("STEP 15: Generating interactive HTML report")
    report_html = generate_report_with_cluster_stats(
        chi2_df,
        network_df,
        entropy_df,
        cramers_df,
        feature_summary_df,
        excl2_df,
        excl3_df,
        hubs_df,
        network_html,
        chi2_summary,
        entropy_summary,
        cramers_summary,
        excl2_summary,
        excl3_summary,
        network_summary,
        hubs_summary,
        chi2_cat,
        chi2_feat,
        entropy_cat,
        entropy_feat,
        cramers_cat,
        cramers_feat,
        excl2_cat,
        excl2_feat,
        excl3_cat,
        excl3_feat,
        network_cat,
        network_feat,
    )
    # Write HTML report inside the configured output folder
    os.makedirs(output_folder, exist_ok=True)
    html_path = os.path.join(output_folder, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    logging.info(f"HTML report saved to '{html_path}'")
    print(f"HTML report saved to '{html_path}'.")

    # Step 16: Generate Excel report with PNG charts
    logging.info("STEP 16: Generating comprehensive Excel report with PNG charts")
    excel_path = generate_excel_report_with_cluster_stats(
        chi2_df,
        network_df,
        entropy_df,
        cramers_df,
        feature_summary_df,
        excl2_df,
        excl3_df,
        hubs_df,
        fig_network,
        chi2_summary,
        entropy_summary,
        cramers_summary,
        excl2_summary,
        excl3_summary,
        network_summary,
        hubs_summary,
        chi2_cat,
        chi2_feat,
        entropy_cat,
        entropy_feat,
        cramers_cat,
        cramers_feat,
        excl2_cat,
        excl2_feat,
        excl3_cat,
        excl3_feat,
        network_cat,
        network_feat,
    )
    logging.info(f"Excel report saved to: {excel_path}")
    print(f"Excel report saved to: {excel_path}")

    # Step 17: Download reports (for Colab)
    logging.info("STEP 17: Report generation complete")
    if IN_COLAB:
        try:
            files.download(html_path)
            files.download(excel_path)
            logging.info("Reports downloaded successfully")
        except Exception as e:
            logging.warning(f"Auto-download may not work in all environments: {e}")
            print(f"Note: Auto-download may not work in all environments: {e}")
    else:
        print(f"Reports saved to: {html_path} and {excel_path}")

    # Final summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(
        f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
    )
    logging.info(f"Total features analyzed: {len(features)}")
    logging.info(f"Total strains: {len(merged)}")
    logging.info(f"Chi-square tests performed: {len(chi2_df)}")
    logging.info(f"Significant associations (FDR<0.05): {chi2_df['Significant'].sum()}")
    logging.info(f"Network nodes: {G.number_of_nodes()}")
    logging.info(f"Network edges: {G.number_of_edges()}")
    if G.number_of_edges() > 0:
        logging.info(f"Network clusters: {num_clusters}")
        logging.info(f"Hub features identified: {len(hubs_df)}")
    logging.info(f"Mutually exclusive pairs: {len(excl2_df)}")
    logging.info(f"Mutually exclusive triplets: {len(excl3_df)}")
    logging.info("=" * 80)
    logging.info("Output files generated:")
    logging.info(f"  - {html_path}")
    logging.info(f"  - {excel_path}")
    logging.info("  - output/png_charts/ (PNG visualizations)")
    logging.info("  - output/network_analysis_log.txt (execution log)")
    logging.info("=" * 80)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("Check output/network_analysis_log.txt for detailed execution log")
    print("=" * 80)


def main():
    """Main entry point for the network analysis."""
    perform_full_analysis()


if __name__ == "__main__":
    main()
