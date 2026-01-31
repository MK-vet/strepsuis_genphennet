# Performance Validation Report - strepsuis-genphennet

**Generated:** 2026-01-31T10:30:00.452314

---

## Benchmark Summary

| Operation | Data Size | Time (ms) | Throughput |
|-----------|-----------|-----------|------------|
| Network Construction (f=10) | 45 pairs | 89.84 | 501 pairs/s |
| Network Construction (f=20) | 190 pairs | 892.2 | 213 pairs/s |
| Network Construction (f=50) | 1225 pairs | 2731.08 | 449 pairs/s |
| Degree Centrality (f=20) | 20 nodes | 729.81 | 27 nodes/s |
| Degree Centrality (f=50) | 50 nodes | 6954.71 | 7 nodes/s |
| Degree Centrality (f=100) | 100 nodes | 18795.13 | 5 nodes/s |
| Community Detection (f=20) | 20 nodes | 1.55 | 12876 nodes/s |
| Community Detection (f=50) | 50 nodes | 0.26 | 195848 nodes/s |
| Community Detection (f=100) | 100 nodes | 0.68 | 146177 nodes/s |
| FDR Correction (n=100) | 100 tests | 0.06 | 1697793 tests/s |
| FDR Correction (n=500) | 500 tests | 0.03 | 15015006 tests/s |
| FDR Correction (n=1000) | 1000 tests | 0.04 | 26666671 tests/s |
| FDR Correction (n=5000) | 5000 tests | 0.26 | 18875049 tests/s |
| Mutual Information (n=50) | 190 pairs | 69.75 | 2724 pairs/s |
| Mutual Information (n=100) | 190 pairs | 73.97 | 2569 pairs/s |
| Mutual Information (n=200) | 190 pairs | 70.19 | 2707 pairs/s |
| Mutual Information (n=500) | 190 pairs | 77.86 | 2440 pairs/s |

---

## Scalability Analysis

### Network Construction

| Data Size | Time (ms) |
|-----------|----------|
| 10 | 89.84 |
| 20 | 892.2 |
| 50 | 2731.08 |

### Degree Centrality

| Data Size | Time (ms) |
|-----------|----------|
| 20 | 729.81 |
| 50 | 6954.71 |
| 100 | 18795.13 |

### Community Detection

| Data Size | Time (ms) |
|-----------|----------|
| 20 | 1.55 |
| 50 | 0.26 |
| 100 | 0.68 |

### FDR Correction

| Data Size | Time (ms) |
|-----------|----------|
| 100 | 0.06 |
| 500 | 0.03 |
| 1000 | 0.04 |
| 5000 | 0.26 |

### Mutual Information

| Data Size | Time (ms) |
|-----------|----------|
| 50 | 69.75 |
| 100 | 73.97 |
| 200 | 70.19 |
| 500 | 77.86 |

