"""
This script performs preprocessing and clustering analysis of single-cell RNA-seq data,
including quality control, doublet removal, normalization, highly variable gene selection,
Harmony-based batch correction, Leiden clustering, UMAP visualization, and marker gene analysis.

Input:
    AnnData object in .h5ad format

Output:
    Processed .h5ad files for each major analysis stage
"""

import re
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import scanpy.external as sce

np.random.seed(0)


def run_scRNAseq_preprocessing(
    input_h5ad,
    output_dir,
    sample_key="sample",
    scrublet_threshold=0.25,
    hvg_n_top_genes=1500,
    harmony_theta=2.5,
    leiden_resolution=2.0,
):
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(r"^HB[^(P)]", regex=True)

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True,
    )

    min_cells_0_1pct = int(np.floor(adata.n_obs * 0.001)) + 1
    sc.pp.filter_genes(adata, min_cells=min_cells_0_1pct)

    adata = adata[
        (adata.obs["total_counts"] >= 1000)
        & (adata.obs["total_counts"] <= 25000)
        & (adata.obs["n_genes_by_counts"] >= 500)
        & (adata.obs["n_genes_by_counts"] <= 5000)
        & (adata.obs["pct_counts_mt"] <= 10),
        :
    ].copy()

    adata = adata[adata.obs["pct_counts_ribo"] < 50, :].copy()
    adata = adata[adata.obs["pct_counts_hb"] < 5, :].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    adata.write(output_dir / "adata_qc_filtered.h5ad")

    sc.pp.scrublet(adata, batch_key=sample_key, verbose=True)
    adata.obs["predicted_doublet"] = adata.obs["doublet_score"] > scrublet_threshold

    n_doublets = int(adata.obs["predicted_doublet"].sum())
    print(f"Number of predicted doublets (score > {scrublet_threshold}): {n_doublets}")

    adata.write(output_dir / "adata_with_doublet_scores.h5ad")

    adata = adata[~adata.obs["predicted_doublet"]].copy()
    adata.write(output_dir / "adata_doublets_removed.h5ad")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=hvg_n_top_genes,
        batch_key=sample_key,
    )

    hvg_genes = adata.var_names[adata.var["highly_variable"]]

    exclude_pattern = re.compile(r"^(MT-|mt-|RPS|RPL)", re.IGNORECASE)
    filtered_hvg_genes = [
        gene for gene in hvg_genes
        if not exclude_pattern.match(gene)
    ]

    adata_hvg = adata[:, filtered_hvg_genes].copy()
    adata_hvg.write(output_dir / "adata_hvg_filtered.h5ad")

    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=20)

    sce.pp.harmony_integrate(
        adata_hvg,
        key=sample_key,
        theta=harmony_theta,
    )

    adata_hvg.write(output_dir / "adata_hvg_harmony.h5ad")

    sc.pp.neighbors(
        adata_hvg,
        use_rep="X_pca_harmony",
        n_pcs=20,
        random_state=0,
    )

    leiden_key = f"leiden_{leiden_resolution}"

    sc.tl.leiden(
        adata_hvg,
        resolution=leiden_resolution,
        key_added=leiden_key,
        flavor="igraph",
        n_iterations=10,
        random_state=0,
    )

    sc.tl.umap(adata_hvg, random_state=0)

    warnings.filterwarnings("ignore", message=".*fragmented.*")
    adata_hvg.X = adata.raw[:, adata_hvg.var_names].X

    sc.tl.rank_genes_groups(
        adata_hvg,
        groupby=leiden_key,
        method="wilcoxon",
        key_added=f"rank_genes_{leiden_key}",
        use_raw=False,
    )

    adata_hvg.write(output_dir / "adata_hvg_clustered_ranked.h5ad")

    return adata_hvg


if __name__ == "__main__":
    input_file = "input_data.h5ad"
    output_directory = "results"

    adata_result = run_scRNAseq_preprocessing(
        input_h5ad=input_file,
        output_dir=output_directory,
        sample_key="sample",
        scrublet_threshold=0.25,
        hvg_n_top_genes=1500,
        harmony_theta=2.5,
        leiden_resolution=2.0,
    )
