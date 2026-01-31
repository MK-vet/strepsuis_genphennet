# Algorithms Documentation

This document provides detailed algorithmic descriptions and Big-O complexity analysis
for the key computational methods in the StrepSuis-GenPhenNet module.

## Overview

All algorithms in the StrepSuis-GenPhenNet module are designed with:
- **Reproducibility**: Fixed random seeds for deterministic results
- **Numerical stability**: Careful handling of edge cases and numerical precision
- **Scalability**: Efficient implementations suitable for datasets up to 10,000+ strains

## 1. Statistical Association Testing

### Chi-Square Test with Test Selection

**Purpose**: Test for independence between binary variables with appropriate test selection.

**Algorithm**:
```
FUNCTION safe_contingency(table):
    INPUT: 2×2 contingency table [[a, b], [c, d]]
    OUTPUT: (chi2_statistic, p_value, phi_coefficient)
    
    PROCEDURE:
        IF table.shape != (2, 2) OR total == 0:
            RETURN (NaN, NaN, NaN)
        
        # Calculate phi coefficient
        (a, b), (c, d) = table.values
        row_sums = [a+b, c+d]
        col_sums = [a+c, b+d]
        total = a + b + c + d
        
        num = a*d - b*c
        den = sqrt(row_sums[0] * row_sums[1] * col_sums[0] * col_sums[1])
        phi = num / den if den > 0 else NaN
        
        # Calculate expected counts
        expected = outer(row_sums, col_sums) / total
        min_expected = min(expected)
        pct_above_5 = count(expected >= 5) / 4
        
        # Cochran's rule for test selection
        IF min_expected < 1 OR pct_above_5 < 0.8:
            # Use Fisher's exact test
            _, p_val = fisher_exact(table)
            chi2 = phi² × total  # Derived for consistency
        ELSE:
            # Use chi-square test
            chi2, p_val = chi2_contingency(table)
        
        RETURN (chi2, p_val, phi)
```

**Complexity**:
- Time: O(1) for 2×2 tables, O(n!) worst case for Fisher's exact
- Space: O(1)

## 2. FDR Correction (Benjamini-Hochberg)

### Purpose
Control False Discovery Rate when testing multiple hypotheses.

### Algorithm
```
FUNCTION fdr_bh_correction(p_values, alpha=0.05):
    INPUT:
        p_values: Array of m raw p-values
        alpha: Target FDR level
    
    OUTPUT:
        reject: Boolean array of rejection decisions
        corrected: Adjusted p-values
    
    PROCEDURE:
        m = length(p_values)
        sorted_idx = argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate BH critical values
        critical = [(i+1) / m * alpha for i in range(m)]
        
        # Find largest k where p_k <= critical_k
        k = max([i for i in range(m) if sorted_p[i] <= critical[i]], default=-1)
        
        # Reject all hypotheses with index <= k
        reject = array of False, length m
        IF k >= 0:
            reject[sorted_idx[:k+1]] = True
        
        # Calculate adjusted p-values
        # Adjusted p_i = min(p_i * m / i, 1)
        # Ensure monotonicity: adj_p[i] >= adj_p[i-1]
        corrected = empty array of size m
        corrected[sorted_idx[-1]] = min(sorted_p[-1], 1.0)
        FOR i = m-2 TO 0:
            adj = min(sorted_p[i] * m / (i+1), corrected[sorted_idx[i+1]])
            corrected[sorted_idx[i]] = adj
        
        RETURN (reject, corrected)
```

**Complexity**:
- Time: O(m log m) dominated by sorting
- Space: O(m)

## 3. Entropy and Mutual Information

### Purpose
Quantify information content and shared information between variables.

### Algorithms
```
FUNCTION entropy(X):
    # Shannon entropy for binary variable
    p = mean(X)  # Probability of 1
    q = 1 - p    # Probability of 0
    
    IF p == 0 OR q == 0:
        RETURN 0.0
    
    RETURN -p * log2(p) - q * log2(q)

FUNCTION joint_entropy(X, Y):
    # Joint entropy for two binary variables
    p11 = mean(X == 1 AND Y == 1)
    p10 = mean(X == 1 AND Y == 0)
    p01 = mean(X == 0 AND Y == 1)
    p00 = mean(X == 0 AND Y == 0)
    
    H = 0
    FOR p in [p11, p10, p01, p00]:
        IF p > 0:
            H -= p * log2(p)
    
    RETURN H

FUNCTION mutual_information(X, Y):
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    RETURN entropy(X) + entropy(Y) - joint_entropy(X, Y)

FUNCTION normalized_mutual_information(X, Y):
    # NMI = 2 * I(X;Y) / (H(X) + H(Y))
    mi = mutual_information(X, Y)
    h_x = entropy(X)
    h_y = entropy(Y)
    
    IF h_x + h_y == 0:
        RETURN 0.0
    
    RETURN 2 * mi / (h_x + h_y)
```

**Complexity**:
- Time: O(n) per pair
- Space: O(1)

## 4. Network Construction

### Purpose
Build co-occurrence networks from significant associations.

### Algorithm
```
FUNCTION build_genotype_phenotype_network(data, pheno_cols, gene_cols, alpha=0.05):
    INPUT:
        data: Combined phenotype/genotype DataFrame
        pheno_cols, gene_cols: Column name lists
        alpha: Significance threshold
    
    OUTPUT:
        G: NetworkX graph
    
    PROCEDURE:
        all_cols = pheno_cols + gene_cols
        G = empty Graph()
        
        # Add nodes with type attribute
        FOR p in pheno_cols:
            G.add_node(p, node_type='Phenotype')
        FOR g in gene_cols:
            G.add_node(g, node_type='Gene')
        
        # Calculate all pairwise associations
        combos = []
        FOR (c1, c2) in combinations(all_cols, 2):
            table = crosstab(data[c1], data[c2])
            chi2, p_val, phi = safe_contingency(table)
            IF p_val is not NaN:
                combos.append((c1, c2, phi, p_val))
        
        # Apply FDR correction
        raw_pvals = [c[3] for c in combos]
        reject, corrected = multipletests(raw_pvals, alpha, 'fdr_bh')
        
        # Add significant edges
        FOR (c1, c2, phi, p_raw), p_corr, is_sig in zip(combos, corrected, reject):
            IF is_sig:
                edge_type = determine_edge_type(c1, c2, pheno_cols, gene_cols)
                G.add_edge(c1, c2, 
                           phi=phi, 
                           weight=abs(phi),
                           pvalue=p_corr, 
                           edge_type=edge_type)
        
        # Remove isolated nodes
        G.remove_nodes(list(isolates(G)))
        
        RETURN G
```

**Complexity**:
- Time: O(m² × n) for contingency tables + O(m² log m²) for FDR
- Space: O(m² + n_edges)

## 5. Community Detection (Louvain)

### Purpose
Identify communities in genotype-phenotype association networks.

### Algorithm
```
FUNCTION louvain_communities(graph, resolution=1.0):
    INPUT:
        graph: NetworkX graph with weighted edges
        resolution: Resolution parameter for modularity
    
    OUTPUT:
        communities: List of sets of nodes
    
    PROCEDURE:
        # Phase 1: Local optimization
        partition = {node: {node} for node in graph.nodes}
        
        REPEAT:
            improvement = False
            FOR EACH node in graph.nodes:
                current_community = partition[node]
                best_community = current_community
                best_delta_Q = 0
                
                # Try moving to neighboring communities
                FOR neighbor in graph.neighbors(node):
                    neighbor_community = partition[neighbor]
                    delta_Q = modularity_gain(node, neighbor_community)
                    
                    IF delta_Q > best_delta_Q:
                        best_delta_Q = delta_Q
                        best_community = neighbor_community
                
                IF best_community != current_community:
                    move(node, current_community, best_community)
                    improvement = True
        
        UNTIL NOT improvement
        
        # Phase 2: Aggregate network and repeat
        IF number_of_communities > 1:
            aggregated_graph = aggregate(graph, partition)
            sub_communities = louvain_communities(aggregated_graph)
            RETURN disaggregate(sub_communities, partition)
        ELSE:
            RETURN list(partition.values())
```

**Modularity**:
Q = (1/2m) Σ[A_ij - k_i×k_j/(2m)] δ(c_i, c_j)

**Complexity**:
- Time: O(n log n) average case for sparse networks
- Space: O(n + e)

## 6. Network Centrality Metrics

### Purpose
Identify important nodes in the association network.

### Algorithms
```
FUNCTION degree_centrality(G):
    # Normalized degree
    n = len(G.nodes)
    FOR node in G.nodes:
        centrality[node] = G.degree[node] / (n - 1)
    RETURN centrality

FUNCTION betweenness_centrality(G):
    # Fraction of shortest paths through node
    centrality = zeros(n)
    FOR s in G.nodes:
        FOR t in G.nodes:
            IF s != t:
                paths = all_shortest_paths(s, t)
                FOR node in G.nodes:
                    IF node != s AND node != t:
                        paths_through = count(paths passing through node)
                        centrality[node] += paths_through / len(paths)
    
    # Normalize
    centrality /= ((n-1) * (n-2) / 2)
    RETURN centrality

FUNCTION eigenvector_centrality(G, max_iter=100):
    # Centrality based on neighbor importance
    x = ones(n) / n  # Initial guess
    
    FOR iter = 1 TO max_iter:
        x_new = A @ x  # Matrix-vector product
        x_new /= norm(x_new)  # Normalize
        
        IF norm(x_new - x) < 1e-6:
            BREAK
        x = x_new
    
    RETURN x
```

**Complexity**:
- Degree: O(n)
- Betweenness: O(n × e) for sparse graphs
- Eigenvector: O(n² × max_iter)

## 7. Cramér's V Effect Size

### Purpose
Measure association strength for contingency tables larger than 2×2.

### Algorithm
```
FUNCTION cramers_v(table):
    chi2, p_value, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    
    # Minimum dimension - 1
    min_dim = min(table.shape) - 1
    
    IF min_dim == 0 OR n == 0:
        RETURN 0.0
    
    # Bias-corrected Cramér's V
    phi2 = chi2 / n
    rows, cols = table.shape
    
    # Bias correction (for small samples)
    phi2_corrected = max(0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    rows_corrected = rows - (rows - 1)² / (n - 1)
    cols_corrected = cols - (cols - 1)² / (n - 1)
    
    RETURN sqrt(phi2_corrected / min(rows_corrected - 1, cols_corrected - 1))
```

**Complexity**:
- Time: O(r × c) for r × c table
- Space: O(r × c)

---

## Scalability Considerations

### Typical Performance

| Operation | 100 strains | 500 strains | 1000 strains |
|-----------|-------------|-------------|--------------|
| Pairwise associations (50 features) | ~2s | ~5s | ~10s |
| Network construction | ~1s | ~2s | ~4s |
| Community detection | ~0.1s | ~0.2s | ~0.5s |
| Centrality metrics | ~0.5s | ~2s | ~5s |
| Full pipeline | ~30s | ~90s | ~180s |

### Memory Optimization

For large datasets (>1000 strains):
1. Sparse representation for binary data
2. Streaming computation for pairwise tests
3. Pruning of non-significant edges during construction

---

## References

1. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289-300.
2. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
3. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *JSTAT*, 2008(10), P10008.
4. Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
5. Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-15
