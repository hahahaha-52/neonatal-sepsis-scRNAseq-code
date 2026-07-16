# Code for neonatal sepsis single-cell RNA-seq analysis

This repository contains the custom analysis scripts used for the manuscript:

**A single-cell transcriptomic atlas reveals the immune panorama of neonatal sepsis**

The scripts support preprocessing, quality control, dimensionality reduction, clustering, cell-type annotation, differential expression analysis, gene-set scoring, trajectory analysis, pathway enrichment, ligand–receptor analysis, and figure generation for the neonatal sepsis single-cell RNA-seq study.

## Reproducibility

This repository is intended to provide the custom code required to reproduce the analyses and figures described in the manuscript. Large input data files are not stored in this repository and should be downloaded from the public data repository associated with the study.

## Repository structure

scripts/
├── 01_preprocessing.py
├── 02_embedding_density.py
├── 03_monocyte_analysis.py
├── 04_innate_like_lymphocyte_analysis.py
├── 05_cd8_t_cell_analysis.py
├── 06_cd4_t_cell_analysis.py
├── 07_b_cell_analysis.py
└── 08_cellphonedb_analysis.py

## Script description

- `01_preprocessing.py`: preprocessing, quality control, normalization and initial data processing
- `02_embedding_density.py`: dimensionality reduction, clustering, UMAP visualization and density analysis
- `03_monocyte_analysis.py`: monocyte subset analysis, differential expression analysis and functional enrichment
- `04_innate_like_lymphocyte_analysis.py`: innate-like lymphocyte subset analysis
- `05_cd8_t_cell_analysis.py`: CD8+ T cell subset analysis, pseudotime analysis, gene-set scoring and enrichment analysis
- `06_cd4_t_cell_analysis.py`: CD4+ T cell subset analysis, pseudotime analysis, gene-set scoring and enrichment analysis
- `07_b_cell_analysis.py`: B cell subset analysis, immunoglobulin gene expression analysis, pseudotime analysis and enrichment analysis
- `08_cellphonedb_analysis.py`: ligand–receptor and cell–cell communication analysis using CellPhoneDB

## Data availability

This repository contains custom analysis code only.

The single-cell RNA-seq data are deposited in a public repository as described in the manuscript. Patient-level clinical information and other restricted metadata are not included in this repository.

Input files, including processed AnnData objects, gene-set tables, cell-type annotation tables and color maps, should be obtained or generated according to the procedures described in the manuscript.

## Requirements

The main Python dependencies are listed in `requirements.txt`.

To install the required packages:

```bash
pip install -r requirements.txt
