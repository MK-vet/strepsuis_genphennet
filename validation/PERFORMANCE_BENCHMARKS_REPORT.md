# Performance Benchmarks Report - strepsuis-genphennet

**Generated:** 2026-01-31T10:30:00.454255
**Total Benchmarks:** 23

---

## Benchmark Results

| Operation | Samples | Features | Time (s) | Throughput (samples/s) |
|-----------|---------|----------|----------|------------------------|
| Chi-Square Testing | 50 | 20 | 0.096 | 518.7 |
| Chi-Square Testing | 100 | 30 | 0.156 | 641.2 |
| Chi-Square Testing | 200 | 40 | 0.186 | 1076.6 |
| FDR Correction | 100 | 1 | 0.000 | 819200.0 |
| FDR Correction | 500 | 1 | 0.000 | 3898052.0 |
| FDR Correction | 1000 | 1 | 0.000 | 7037422.8 |
| Network Construction | 20 | 20 | 0.000 | 42324.0 |
| Network Construction | 50 | 50 | 0.001 | 42573.1 |
| Network Construction | 100 | 100 | 0.005 | 21840.8 |
| Centrality Metrics | 20 | 20 | 0.002 | 12565.3 |
| Centrality Metrics | 50 | 50 | 0.009 | 5346.9 |
| Centrality Metrics | 100 | 100 | 0.031 | 3178.2 |
| Community Detection | 20 | 20 | 0.001 | 38639.4 |
| Community Detection | 50 | 50 | 0.002 | 22482.3 |
| Community Detection | 100 | 100 | 0.006 | 17942.8 |
| Logistic Regression CV | 50 | 10 | 0.009 | 5760.0 |
| Logistic Regression CV | 100 | 20 | 0.010 | 10500.5 |
| Logistic Regression CV | 200 | 30 | 0.009 | 22003.5 |
| Mutual Information | 50 | 20 | 0.009 | 5517.9 |
| Mutual Information | 100 | 30 | 0.012 | 8003.5 |
| Mutual Information | 200 | 40 | 0.017 | 11570.5 |
| Full Pipeline | 50 | 20 | 0.037 | 1352.8 |
| Full Pipeline | 100 | 20 | 0.035 | 2840.6 |

---

## Performance Summary

### Chi-Square Testing

- **Average Throughput:** 745.5 samples/s
- **Scalability:** Tested with 50-200 samples

### FDR Correction

- **Average Throughput:** 3918225.0 samples/s
- **Scalability:** Tested with 100-1000 samples

### Network Construction

- **Average Throughput:** 35579.3 samples/s
- **Scalability:** Tested with 20-100 samples

### Centrality Metrics

- **Average Throughput:** 7030.1 samples/s
- **Scalability:** Tested with 20-100 samples

### Community Detection

- **Average Throughput:** 26354.8 samples/s
- **Scalability:** Tested with 20-100 samples

### Logistic Regression CV

- **Average Throughput:** 12754.6 samples/s
- **Scalability:** Tested with 50-200 samples

### Mutual Information

- **Average Throughput:** 8364.0 samples/s
- **Scalability:** Tested with 50-200 samples

### Full Pipeline

- **Average Throughput:** 2096.7 samples/s
- **Scalability:** Tested with 50-100 samples

