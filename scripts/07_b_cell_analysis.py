#!/usr/bin/env python3

"""
B cell subcluster analysis for neonatal sepsis scRNA-seq data.

This script performs:
1. B cell marker dotplot
2. Diffusion map, DPT pseudotime and PAGA analysis
3. Pseudotime density ridge plot
4. Immunoglobulin gene expression composition in plasma cells
5. Immunoglobulin gene expression heatmap
6. Differential expression analysis across conditions
7. GO enrichment analysis for significant DEGs

Example
-------
python scripts/08_b_cell_analysis.py \
    --input data/adata_10_celltype_Subset_log.h5ad \
    --color-map data/celltype_color_map.csv \
    --outdir results/B_cell
"""

import argparse
import re
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


np.random.seed(0)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def make_dir(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(path, dpi=300):
    """Save current matplotlib figure and close it."""
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def make_colormaps():
    """Generate custom colormaps."""

    bluegreen_cmap = LinearSegmentedColormap.from_list(
        "bluegreen",
        ["#FFFFFF", "#E2F6EC", "#8FCBB5", "#3D8C90", "#0B3556"]
    )

    pink_purple_cmap = LinearSegmentedColormap.from_list(
        "pink_purple_gradient",
        ["#f8d8e6", "#d18cd2", "#9a58c1"]
    )

    return bluegreen_cmap, pink_purple_cmap


def load_celltype_colors(color_map_file):
    """Load cell-type color mapping from CSV."""

    if color_map_file is None:
        return {}

    color_map_file = Path(color_map_file)

    if not color_map_file.exists():
        print(f"Warning: color map file not found: {color_map_file}")
        return {}

    color_df = pd.read_csv(color_map_file)

    if not {"celltype", "color"}.issubset(color_df.columns):
        raise ValueError("Color map file must contain 'celltype' and 'color' columns.")

    return dict(zip(color_df["celltype"], color_df["color"]))


def validate_adata(adata):
    """Validate required AnnData fields."""

    required_obs = [
        "celltype_Major",
        "celltype_Subset",
        "condition"
    ]

    missing_obs = [
        col for col in required_obs
        if col not in adata.obs.columns
    ]

    if missing_obs:
        raise ValueError(f"Missing required columns in adata.obs: {missing_obs}")

    print("AnnData validation passed.")


def make_safe_sheet_name(name, existing_names):
    """Create a valid and unique Excel sheet name."""

    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    sheet_name = name[:31]

    if sheet_name not in existing_names:
        existing_names.add(sheet_name)
        return sheet_name

    for i in range(1, 100):
        suffix = f"_{i}"
        candidate = name[:31 - len(suffix)] + suffix

        if candidate not in existing_names:
            existing_names.add(candidate)
            return candidate

    raise ValueError(f"Could not create unique sheet name for {name}.")


# -----------------------------------------------------------------------------
# Load B cell data
# -----------------------------------------------------------------------------

def load_b_cell_data(input_file):
    """Load AnnData and subset B cells."""

    adata = sc.read_h5ad(input_file)

    validate_adata(adata)

    b_mask = adata.obs["celltype_Major"].astype(str).str.fullmatch("B")

    adata_b = adata[b_mask].copy()

    if adata_b.n_obs == 0:
        raise ValueError("No B cells found in adata.obs['celltype_Major'].")

    print(f"Loaded AnnData: {adata.shape}")
    print(f"B cell subset: {adata_b.shape}")

    return adata, adata_b


# -----------------------------------------------------------------------------
# Marker dotplot
# -----------------------------------------------------------------------------

def plot_b_cell_marker_dotplot(adata_b, outdir, cmap):
    """Plot marker gene dotplot for B cell subtypes."""

    marker_genes = [
        "CD79A", "CD79B", "MS4A1", "FCER2", "IGKC", "IGHD", "IGHM", "BACH2",
        "AFF3", "PLCG2", "MYBL2", "IGLC2", "CD83",
        "TCL1A", "FCRL1", "NEIL1", "PAX5", "FOXP1",
        "MZB1", "JCHAIN"
    ]

    marker_genes = list(dict.fromkeys(marker_genes))
    marker_genes = [gene for gene in marker_genes if gene in adata_b.var_names]

    if len(marker_genes) == 0:
        raise ValueError("None of the B cell marker genes were found in adata_b.var_names.")

    sc.pl.dotplot(
        adata_b,
        var_names=marker_genes,
        groupby="celltype_Subset",
        standard_scale="var",
        dot_max=1.0,
        color_map=cmap,
        show=False,
        use_raw=False
    )

    plt.gcf().set_size_inches(8, 4.5)

    save_figure(
        Path(outdir) / "1.1.3_b_cell_subtypes_dotplot.pdf"
    )


# -----------------------------------------------------------------------------
# Diffusion map, DPT pseudotime and PAGA
# -----------------------------------------------------------------------------

def run_b_cell_pseudotime(
    adata_b,
    outdir,
    root_celltype="B_01_Naive_FCER2_IGKC"
):
    """Run diffusion map, DPT pseudotime and PAGA analysis."""

    if "X_pca_harmony" not in adata_b.obsm:
        raise ValueError(
            "X_pca_harmony was not found in adata_b.obsm. "
            "Please run PCA/Harmony integration before DPT analysis."
        )

    adata_dpt = adata_b.copy()

    sc.pp.neighbors(
        adata_dpt,
        use_rep="X_pca_harmony",
        n_pcs=20,
        n_neighbors=30,
        metric="cosine"
    )

    sc.tl.diffmap(adata_dpt)

    root_cells = np.flatnonzero(
        adata_dpt.obs["celltype_Subset"] == root_celltype
    )

    if len(root_cells) == 0:
        raise ValueError(f"Root cell type was not found: {root_celltype}")

    adata_dpt.uns["iroot"] = root_cells[0]

    sc.tl.dpt(
        adata_dpt,
        n_dcs=10,
        allow_kendall_tau_shift=False
    )

    sc.tl.paga(
        adata_dpt,
        groups="celltype_Subset"
    )

    sc.pl.paga(
        adata_dpt,
        node_size_scale=1.5,
        node_size_power=0.2,
        edge_width_scale=1,
        threshold=0.03,
        color=["dpt_pseudotime"],
        show=False
    )

    plt.tight_layout()

    save_figure(
        Path(outdir) / "1.3.1_paga_b_cell_pseudotime.pdf"
    )

    df_pseudotime = adata_dpt.obs[
        ["celltype_Subset", "dpt_pseudotime"]
    ].dropna().copy()

    df_pseudotime.to_csv(
        Path(outdir) / "1.3.1_b_cell_pseudotime_values.csv",
        index=False
    )

    return adata_dpt, df_pseudotime


def plot_pseudotime_ridge(df_pseudotime, outdir, color_map=None):
    """Plot pseudotime density ridge plot using Python."""

    order = (
        df_pseudotime
        .groupby("celltype_Subset")["dpt_pseudotime"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    x_grid = np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, celltype in enumerate(order):
        values = df_pseudotime.loc[
            df_pseudotime["celltype_Subset"] == celltype,
            "dpt_pseudotime"
        ].dropna().values

        if len(values) < 5:
            print(f"Skipped ridge plot for {celltype}: fewer than 5 cells.")
            continue

        if len(np.unique(values)) < 2:
            print(f"Skipped ridge plot for {celltype}: pseudotime values are constant.")
            continue

        try:
            kde = gaussian_kde(values)
            density = kde(x_grid)
        except Exception as error:
            print(f"Skipped ridge plot for {celltype}: {error}")
            continue

        if density.max() == 0:
            continue

        density = density / density.max() * 0.8

        color = "#96C0FF"
        if color_map is not None and celltype in color_map:
            color = color_map[celltype]

        ax.fill_between(
            x_grid,
            i,
            i + density,
            alpha=0.7,
            color=color,
            linewidth=0
        )

        ax.plot(
            x_grid,
            i + density,
            color="black",
            linewidth=0.3
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("B cell pseudotime distribution")

    plt.tight_layout()

    save_figure(
        Path(outdir) / "1.3.2_b_cell_pseudotime_ridgeplot.pdf"
    )


# -----------------------------------------------------------------------------
# Immunoglobulin gene expression analysis
# -----------------------------------------------------------------------------

def get_ig_fraction(adata_subset, ig_genes):
    """Calculate fraction of mean IG expression in one sub
