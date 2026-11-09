# EarthEM: Embedding-Based Remote Sensing Analysis of Land Use and Climate Dynamics

**Satellite image embeddings for data-efficient geospatial machine learning**

This repository presents a hypothesis-driven investigation into the utility of Google’s 64-dimensional satellite image embeddings (Satellite Embedding V1 / AlphaEarth) for land-cover classification, hydrological monitoring, and urban-change detection. The study focuses on the Krishna Raja Sagara (KRS) Reservoir and surrounding agricultural–urban landscape in Karnataka, India.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Traditional remote-sensing workflows often require large volumes of labeled spectral data and extensive preprocessing. Pre-trained satellite embeddings offer a compact, multi-modal representation of the Earth’s surface that can support data-efficient analysis. This project evaluates whether these 64-dimensional annual embeddings encode sufficient information for three practical tasks under minimal supervision:

1. **Crop-type mapping** with sparse ground-truth labels  
2. **Detection of climatic extremes** (drought) via temporal embedding similarity of a large water body  
3. **Quantification of urban encroachment** into agricultural land through shifts in embedding space

All analyses are performed on publicly available Google Earth Engine data and are designed to be reproducible.

**Authors:** Suyash Gaurav & Aman Kamat  
**Affiliation:** Department of Digital Business and Innovation & Department of Business Economics  
**Contact:** s23224522@al.tiu.ac.jp

---

## Dataset

| Attribute              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Source**             | [GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |
| **Representation**     | 64-dimensional spectral–temporal embeddings per 10 m pixel (bands A00–A63) |
| **Temporal coverage**  | Annual composites, 2017–2024                                                |
| **Spatial subset**     | KRS Reservoir and Mysore region, Karnataka, India                           |
| **Advantages**         | Multi-modal (optical + radar heritage), reduced need for raw band engineering, global coverage |

---

## Research Questions & Hypotheses

### RQ1 — Crop-Type Mapping with Minimal Labels
**Question:** Can dominant crop types around the KRS Reservoir be mapped reliably using only sparse labeled samples in embedding space?

**Hypothesis:** A Random Forest classifier trained on ≤150 samples per class in the 64-D embedding space will exceed 80 % overall accuracy when distinguishing rice, sugarcane, other crops, water, and urban areas.

**Key result:** 82.1 % accuracy on a held-out stratified test set. Perfect separation of water and urban classes; moderate confusion among crop types. Top discriminative dimensions include A09, A22, and A28.

### RQ2 — Embedding Similarity as a Proxy for Climatic Extremes
**Question:** Do inter-annual variations in the mean embedding of the KRS water body reflect documented drought years?

**Hypothesis:** Cosine similarity among drought-year embeddings will be significantly lower than similarity among normal or good-monsoon years.

**Key result:** Mean within-group cosine similarity ≈ 0.90 (drought) vs ≈ 0.97 (normal). One-way ANOVA: *F* ≈ 12.12, *p* ≈ 0.012. PCA trajectories show clear divergence of drought years in embedding space.

### RQ3 — Detecting Urban Encroachment via Embedding Shift
**Question:** Has urban expansion around Mysore produced a measurable convergence of agricultural and urban signatures in embedding space between 2017 and 2024?

**Hypothesis:** The separation between urban and agricultural clusters will decrease over time, and 2024 agricultural samples will lie closer to the urban centroid than their 2017 counterparts.

**Key result:** Cluster separation ratio declined by 10.6 %. Agricultural samples moved ≈ 21 % closer to the urban centroid (*t*-test, *p* < 0.0001).

---

## Repository Structure

```
EarthEmbd/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_earth_embeddings_analysis.ipynb   # Complete reproducible analysis
└── src/
    └── earth_embd/
        ├── __init__.py
        ├── ee_utils.py          # Earth Engine helpers
        ├── analysis.py          # Classification, similarity, clustering
        └── visualization.py     # Plotting utilities
```

---

## Installation

```bash
git clone https://github.com/Suyash-Gaurav/EarthEmbd.git
cd EarthEmbd
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Google Earth Engine Authentication

```python
import ee
ee.Authenticate()                  # Follow the interactive flow once
ee.Initialize(project="YOUR-GEE-PROJECT-ID")
```

Replace `YOUR-GEE-PROJECT-ID` with your own Google Cloud / Earth Engine project. Never commit project IDs or credentials.

---

## How to Run

1. Authenticate and initialize Earth Engine (see above).  
2. Open the analysis notebook:

```bash
jupyter notebook notebooks/01_earth_embeddings_analysis.ipynb
```

or

```bash
jupyter lab notebooks/01_earth_embeddings_analysis.ipynb
```

3. Execute cells sequentially. The notebook is structured to mirror a scientific paper: Setup → Data → RQ1 → RQ2 → RQ3 → Summary.

---

## Methods Summary

| Component              | Technique                                      |
|------------------------|------------------------------------------------|
| Platform               | Google Earth Engine (Python API)               |
| Classifier (RQ1)       | Random Forest (100 trees, stratified 70/30 split) |
| Similarity (RQ2)       | Cosine similarity + one-way ANOVA              |
| Change detection (RQ3) | K-Means (k=2) + centroid distance / separation ratio + *t*-test |
| Dimensionality reduction | PCA for visualization and interpretability   |
| Visualization          | Matplotlib + Seaborn                           |

All sampling is performed at 100 m scale to balance computational cost and spatial detail. Labeled points for RQ1 were manually curated from 2024 imagery and expanded via buffered random sampling.

---

## Key Results at a Glance

| RQ  | Metric                          | Value          | Supported? |
|-----|----------------------------------|----------------|------------|
| 1   | Overall accuracy                 | 82.1 %         | Yes        |
| 2   | ANOVA *p*-value (climate groups) | ≈ 0.012        | Yes        |
| 2   | Drought vs normal similarity     | 0.90 vs 0.97   | Yes        |
| 3   | Separation ratio change          | −10.6 %        | Yes        |
| 3   | Agri → urban distance reduction  | ≈ 21 %         | Yes        |

---

## Reproducibility Notes

- Random seeds are fixed (`random_state=42`) wherever applicable.  
- Earth Engine sampling can exhibit minor stochastic variation; results are expected to be stable within a few percentage points.  
- The notebook contains explicit geometry definitions for the study regions so that anyone with Earth Engine access can reproduce the analysis.  
- No proprietary data or private assets are required.

---

## Citation

If you use this code or findings in academic work, please cite the Google Satellite Embedding dataset and acknowledge the authors of this repository.

```
Gaurav, S. & Kamat, A. (2025). EarthEM: Embedding-Based Remote Sensing Analysis of Land Use and Climate Dynamics.
https://github.com/Suyash-Gaurav/EarthEmbd
```

Dataset:  
Gorelick et al. / Google Earth Engine – Satellite Embedding V1 (Annual).

---

## License

MIT License – free for academic and non-commercial research use. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Google Earth Engine team for the Satellite Embedding V1 dataset  
- Open-source scientific Python ecosystem (scikit-learn, SciPy, NumPy, Pandas, Matplotlib, Seaborn)
