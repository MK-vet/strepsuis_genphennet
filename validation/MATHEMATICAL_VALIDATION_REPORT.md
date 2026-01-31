# Mathematical Validation Report - strepsuis-genphennet

**Generated:** 2026-01-31T10:30:00.456493
**Total Tests:** 17
**Passed:** 17
**Coverage:** 100.0%

---

## Test Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Chi-Square vs scipy | chi2>0, 0≤p≤1 | chi2=16.67, p=0.0000 | ✅ PASS |
| Fisher's Exact vs scipy | OR>0, 0≤p≤1 | OR=6.00, p=0.5238 | ✅ PASS |
| Phi Perfect Positive | 1.0 | 1.0000 | ✅ PASS |
| Phi No Association | ~0.0 | 0.0000 | ✅ PASS |
| FDR vs statsmodels | reject>0 | reject=2 | ✅ PASS |
| FDR Monotonicity | Monotonically non-decreasing | Monotonic | ✅ PASS |
| FDR Control | ≤5.0% | 0.0% | ✅ PASS |
| Degree Centrality | B > A | B=1.00, A=0.33 | ✅ PASS |
| Betweenness Centrality | C ≥ A | C=0.67, A=0.00 | ✅ PASS |
| Community Detection | ≥2 communities | 2 communities | ✅ PASS |
| Conditional Probability | P(B|A) > P(B|¬A) | P(B|A)=0.82 > P(B|¬A)=0.21 | ✅ PASS |
| Mutual Information | MI(A,B) > MI(A,C) | MI(A,B)=0.200 > MI(A,C)=0.002 | ✅ PASS |
| Cross-Validation | Accuracy > 0.5 | Accuracy = 0.820 | ✅ PASS |
| AUC Calculation | AUC = 1.0 for perfect | AUC = 1.000 | ✅ PASS |
| Bootstrap Coverage | ~95% | 96.0% | ✅ PASS |
| Triangle Counting | 2 triangles | 2 triangles | ✅ PASS |
| Clustering Coefficient | CC = 1.0 for complete graph | CC = 1.000 | ✅ PASS |

---

## Detailed Results

### Chi-Square vs scipy - ✅ PASS

- **Expected:** chi2>0, 0≤p≤1
- **Actual:** chi2=16.67, p=0.0000
- **Details:** Should produce valid chi-square

### Fisher's Exact vs scipy - ✅ PASS

- **Expected:** OR>0, 0≤p≤1
- **Actual:** OR=6.00, p=0.5238
- **Details:** Should produce valid Fisher's exact

### Phi Perfect Positive - ✅ PASS

- **Expected:** 1.0
- **Actual:** 1.0000
- **Details:** Perfect positive association

### Phi No Association - ✅ PASS

- **Expected:** ~0.0
- **Actual:** 0.0000
- **Details:** Independent variables

### FDR vs statsmodels - ✅ PASS

- **Expected:** reject>0
- **Actual:** reject=2
- **Details:** FDR correction should work

### FDR Monotonicity - ✅ PASS

- **Expected:** Monotonically non-decreasing
- **Actual:** Monotonic
- **Details:** Corrected p-values should be monotonic

### FDR Control - ✅ PASS

- **Expected:** ≤5.0%
- **Actual:** 0.0%
- **Details:** FDR should be controlled

### Degree Centrality - ✅ PASS

- **Expected:** B > A
- **Actual:** B=1.00, A=0.33
- **Details:** Hub node should have highest centrality

### Betweenness Centrality - ✅ PASS

- **Expected:** C ≥ A
- **Actual:** C=0.67, A=0.00
- **Details:** Bridge node should have high betweenness

### Community Detection - ✅ PASS

- **Expected:** ≥2 communities
- **Actual:** 2 communities
- **Details:** Should detect planted communities

### Conditional Probability - ✅ PASS

- **Expected:** P(B|A) > P(B|¬A)
- **Actual:** P(B|A)=0.82 > P(B|¬A)=0.21
- **Details:** Causal relationship detected

### Mutual Information - ✅ PASS

- **Expected:** MI(A,B) > MI(A,C)
- **Actual:** MI(A,B)=0.200 > MI(A,C)=0.002
- **Details:** Correlated variables have higher MI

### Cross-Validation - ✅ PASS

- **Expected:** Accuracy > 0.5
- **Actual:** Accuracy = 0.820
- **Details:** Should achieve better than random

### AUC Calculation - ✅ PASS

- **Expected:** AUC = 1.0 for perfect
- **Actual:** AUC = 1.000
- **Details:** Perfect predictions should have AUC=1

### Bootstrap Coverage - ✅ PASS

- **Expected:** ~95%
- **Actual:** 96.0%
- **Details:** CI should contain true value ~95%

### Triangle Counting - ✅ PASS

- **Expected:** 2 triangles
- **Actual:** 2 triangles
- **Details:** Should count triangles correctly

### Clustering Coefficient - ✅ PASS

- **Expected:** CC = 1.0 for complete graph
- **Actual:** CC = 1.000
- **Details:** Complete graph should have CC=1

