# Real Data Validation Report - strepsuis-genphennet

**Generated:** 2026-01-31T10:30:00.450148
**Data Source:** S. suis strains (AMR_genes.csv, MIC.csv, Virulence.csv)
**Total Tests:** 8
**Passed:** 8
**Coverage:** 100.0%

---

## Statistical Validation Results

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Strain Count | ≥50 strains | 91 strains | ✅ PASS |
| Association Network Construction | Network built | 1 significant edges | ✅ PASS |
| Network Density | [0, 1] | 0.222 | ✅ PASS |
| Degree Centrality | Valid degrees | Hub: ant(6)-Ia (degree=6) | ✅ PASS |
| Community Detection | ≥1 community | 7 communities | ✅ PASS |
| Fisher's Exact Test | Valid OR and p-value | OR=1.00, p=1.0000 | ✅ PASS |
| FDR Correction | Correction applied | 0/20 significant | ✅ PASS |
| Mutual Information | MI ≥ 0 | MI = 0.0000 | ✅ PASS |

---

## Biological Validation Results

### Genotype-Phenotype Network

**Description:** Network of significant associations between AMR genes and resistance phenotypes

**Result:** 1 significant edges (p < 0.05)

**Interpretation:** Dense networks suggest strong genotype-phenotype relationships. Sparse networks may indicate complex resistance mechanisms.

### Network Density

**Description:** Proportion of possible edges that are significant

**Result:** Density: 0.222 (10 edges)

**Interpretation:** High density (>0.3) suggests extensive gene co-selection. Low density indicates independent evolution.

### Hub Gene Identification

**Description:** Gene with most significant associations (highest degree)

**Result:** Hub gene: ant(6)-Ia with 6 connections

**Interpretation:** Hub genes may be key drivers of resistance phenotypes or located on mobile genetic elements.

### Gene Communities

**Description:** Groups of co-occurring genes detected by hierarchical clustering

**Result:** 7 gene communities identified

**Interpretation:** Communities may represent resistance islands, plasmids, or functionally related gene clusters.

### AMR-Virulence Odds Ratio

**Description:** Odds ratio from Fisher's exact test

**Result:** OR=1.00, p=1.0000

**Interpretation:** OR>1 suggests positive association. OR<1 suggests negative association.

### Multiple Testing Correction

**Description:** FDR-corrected significant associations

**Result:** 0 associations significant after FDR correction

**Interpretation:** FDR correction reduces false positives while maintaining statistical power.

### Mutual Information

**Description:** Information shared between AMR gene and virulence factor

**Result:** MI = 0.0000

**Interpretation:** Higher MI indicates stronger dependency. MI=0 indicates independence.

