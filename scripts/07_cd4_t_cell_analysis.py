#!/usr/bin/env python3

"""
CD4+ T cell subcluster analysis for neonatal sepsis scRNA-seq data.

This script performs:
1. CD4+ T cell marker dotplot
2. Diffusion map, DPT pseudotime and PAGA analysis
3. Pseudotime density ridge plot
4. Differential expression analysis across conditions
5. GO enrichment analysis for significant DEGs
6. Functional gene-set scoring and UMAP visualization

Example
-------
python scripts/07_cd4_t_cell_analysis.py \
    --input data/adata_10_celltype_Subset_log.h5ad \
    --gene-sets data/exhaustion_cytotoxic_IFN_genes.csv \
    --color-map data/celltype_color_map.csv \
    --outdir results/CD4T
"""

import argparse
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

    green_purple_cmap = LinearSegmentedColormap.from_list(
        "green_purple",
        ["#0E5D29", "#f7f7f7", "#40004C"]
    )

    pink_purple_cmap = LinearSegmentedColormap.from_list(
        "pink_purple_gradient",
        ["#f8d8e6", "#d18cd2", "#9a58c1"]
    )

    return bluegreen_cmap, green_purple_cmap, pink_purple_cmap


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


def get_score_limits(adata, score_name, lower=1, upper=99):
    """Use percentile-based limits for continuous UMAP coloring."""

    values = adata.obs[score_name].dropna().values

    if len(values) == 0:
        return None, None

    vmin = np.percentile(values, lower)
    vmax = np.percentile(values, upper)

    if vmin == vmax:
        vmin = np.min(values)
        vmax = np.max(values)

    return vmin, vmax


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

def load_cd4_data(input_file):
    """Load AnnData and subset CD4+ T cells."""

    adata = sc.read_h5ad(input_file)

    adata_cd4 = adata[
        adata.obs["celltype_Major"].isin(["CD4T"])
    ].copy()

    print(f"Loaded AnnData: {adata.shape}")
    print(f"CD4+ T cell subset: {adata_cd4.shape}")

    return adata, adata_cd4


# -----------------------------------------------------------------------------
# Marker dotplot
# -----------------------------------------------------------------------------

def plot_cd4_marker_dotplot(adata_cd4, outdir, cmap):
    """Plot marker gene dotplot for CD4+ T cell subtypes."""

    marker_genes = [
        "CD3E", "CD4", "CD40LG",
        "CCR7", "LEF1", "TCF7", "IL7R",
        "CD27", "SELL", "ANK3", "CAMK4", "PLCL1",
        "S100A4", "GZMA", "GZMM",
        "FOXP3", "IL2RA",
        "ICOS", "GPR183",
        "STAT1", "ISG15", "IFITM1",
        "MKI67", "TMSB10",
        "CCL5"
    ]

    marker_genes = [gene for gene in marker_genes if gene in adata_cd4.var_names]

    sc.pl.dotplot(
        adata_cd4,
        var_names=marker_genes,
        groupby="celltype_Subset",
        standard_scale="var",
        dot_max=0.8,
        color_map=cmap,
        show=False
    )

    plt.gcf().set_size_inches(8.5, 5)

    save_figure(
        Path(outdir) / "1.1.3_cd4_cell_subtypes_dotplot.pdf"
    )


# -----------------------------------------------------------------------------
# Diffusion map, DPT pseudotime and PAGA
# -----------------------------------------------------------------------------

def run_cd4_pseudotime(
    adata_cd4,
    outdir,
    root_celltype="CD4_03_Naive_CCR7_TCF7"
):
    """Run diffusion map, DPT pseudotime and PAGA analysis."""

    adata_dpt = adata_cd4.copy()

    if "X_pca_harmony" not in adata_dpt.obsm:
        raise ValueError(
            "X_pca_harmony was not found in adata.obsm. "
            "Please run PCA/Harmony integration before DPT analysis."
        )

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
        n_branchings=1
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
        Path(outdir) / "1.3.1_paga_cd4_pseudotime.pdf"
    )

    df_pseudotime = adata_dpt.obs[
        ["celltype_Subset", "dpt_pseudotime"]
    ].dropna().copy()

    df_pseudotime.to_csv(
        Path(outdir) / "1.3.1_cd4_pseudotime_values.csv",
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
            continue

        kde = gaussian_kde(values)
        density = kde(x_grid)

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
    ax.set_title("CD4+ T cell pseudotime distribution")

    plt.tight_layout()

    save_figure(
        Path(outdir) / "1.3.2_cd4_pseudotime_ridgeplot.pdf"
    )


# -----------------------------------------------------------------------------
# Differential expression analysis
# -----------------------------------------------------------------------------

def deg_analysis(
    adata_obj,
    group,
    ref,
    groupby="condition",
    p_adj_thresh=0.05,
    logfc_thresh=0.5,
    min_cells=10
):
    """Run differential expression analysis for one comparison."""

    counts = adata_obj.obs[groupby].value_counts()

    if group not in counts.index or ref not in counts.index:
        print(f"Skipped {group} vs {ref}: one group is missing.")
        return None, None

    if counts[group] < min_cells or counts[ref] < min_cells:
        print(
            f"Skipped {group} vs {ref}: too few cells "
            f"({group}: {counts[group]}, {ref}: {counts[ref]})."
        )
        return None, None

    sc.tl.rank_genes_groups(
        adata_obj,
        groupby=groupby,
        groups=[group],
        reference=ref,
        method="wilcoxon",
        use_raw=False
    )

    df_all = sc.get.rank_genes_groups_df(
        adata_obj,
        group=group
    )

    df_sig = df_all[
        (df_all["pvals_adj"] < p_adj_thresh) &
        (df_all["logfoldchanges"] > logfc_thresh)
    ].copy()

    return df_all, df_sig


def make_cd4_subsets(adata_cd4):
    """Generate CD4+ T cell subtype subsets for DEG analysis."""

    subtype_patterns = {
        "naive": "Naive",
        "cMemory": "cMemory",
        "eMemory": "eMemory",
        "Treg": "Treg",
        "Th1": "Th1",
        "Dividing": "Dividing",
        "Activated": "Activated",
        "activation": "Dividing|Activated"
    }

    datasets = {}

    for name, pattern in subtype_patterns.items():
        mask = adata_cd4.obs["celltype_Subset"].astype(str).str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )

        subset = adata_cd4[mask].copy()

        if subset.n_obs == 0:
            print(f"Skipped subset {name}: no cells found.")
            continue

        datasets[name] = subset
        print(f"{name}: {subset.n_obs} cells")

    return datasets


def run_cd4_deg_analysis(adata_cd4, outdir):
    """Run DEG analysis across conditions for CD4+ T cell subtypes."""

    datasets = make_cd4_subsets(adata_cd4)

    comparisons = [
        ("AC", "HC"),
        ("CO", "HC"),
        ("CO", "AC")
    ]

    deg_results_all = {}
    deg_results_sig = {}

    for dataset_name, adata_obj in datasets.items():
        print(f"\nRunning DEG analysis for: {dataset_name}")

        for group, ref in comparisons:
            comp_name = f"{dataset_name}_{group}_vs_{ref}"

            df_all, df_sig = deg_analysis(
                adata_obj=adata_obj,
                group=group,
                ref=ref
            )

            if df_all is None:
                continue

            deg_results_all[comp_name] = df_all
            deg_results_sig[comp_name] = df_sig

            print(f"{comp_name}: {len(df_sig)} significant DEGs")

    if len(deg_results_all) == 0:
        print("No DEG results were generated.")
        return None

    excel_sig = Path(outdir) / "1.5_CD4T_condition_DEG_sig.xlsx"
    excel_all = Path(outdir) / "1.5_CD4T_condition_DEG_all.xlsx"

    with pd.ExcelWriter(excel_sig, mode="w", engine="openpyxl") as writer:
        for comp_name, df_sig in deg_results_sig.items():
            sheet_name = comp_name[:31]
            df_sig.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(excel_all, mode="w", engine="openpyxl") as writer:
        for comp_name, df_all in deg_results_all.items():
            sheet_name = comp_name[:31]
            df_all.to_excel(writer, sheet_name=sheet_name, index=False)

    print("DEG analysis completed.")

    return excel_sig


# -----------------------------------------------------------------------------
# GO enrichment analysis
# -----------------------------------------------------------------------------

def plot_go_barplot(go_df, topn, title, savepath, cmap):
    """Plot GO enrichment barplot."""

    if go_df is None or go_df.empty:
        print(f"No GO results for {title}")
        return

    df_plot = go_df.sort_values("Adjusted P-value").head(topn).copy()

    df_plot["Adjusted P-value"] = df_plot["Adjusted P-value"].replace(
        0,
        np.nextafter(0, 1)
    )

    df_plot["log10P"] = -np.log10(df_plot["Adjusted P-value"])

    norm = plt.Normalize(
        df_plot["log10P"].min(),
        df_plot["log10P"].max()
    )

    df_plot["color"] = [
        cmap(norm(value)) for value in df_plot["log10P"]
    ]

    plt.figure(figsize=(6, 5))

    sns.barplot(
        data=df_plot,
        x="log10P",
        y="Term",
        hue="Term",
        palette=df_plot.set_index("Term")["color"].to_dict(),
        legend=False
    )

    plt.xlim(0, df_plot["log10P"].max() * 1.05)
    plt.xlabel(r"$-\log_{10}$(adjusted P-value)")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()

    save_figure(savepath)


def run_go_enrichment(
    excel_path,
    outdir,
    cmap,
    organism="Human",
    go_library="GO_Biological_Process_2025"
):
    """Run GO enrichment analysis for DEG Excel sheets."""

    if excel_path is None:
        print("Skipped GO enrichment: no DEG Excel file available.")
        return

    go_outdir = Path(outdir) / "1.5_GO_results"
    make_dir(go_outdir)

    sheet_names = pd.ExcelFile(excel_path).sheet_names

    for sheet in sheet_names:
        print(f"Running GO enrichment for: {sheet}")

        df_sig = pd.read_excel(excel_path, sheet_name=sheet)

        if df_sig.empty:
            print(f"Skipped {sheet}: no significant DEGs.")
            continue

        if "names" not in df_sig.columns:
            print(f"Skipped {sheet}: column 'names' was not found.")
            continue

        genes = df_sig["names"].dropna().unique().tolist()

        if len(genes) < 5:
            print(f"Skipped {sheet}: too few genes.")
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=[go_library],
                organism=organism,
                outdir=None,
                cutoff=1
            )
        except Exception as error:
            print(f"GO enrichment failed for {sheet}: {error}")
            continue

        go_df = enr.results

        save_dir = go_outdir / sheet
        make_dir(save_dir)

        go_df.to_csv(
            save_dir / f"{sheet}_GO_enrichment.csv",
            index=False
        )

        plot_go_barplot(
            go_df=go_df,
            topn=20,
            title=f"{sheet} GO enrichment",
            savepath=save_dir / f"{sheet}_GO_enrichment.pdf",
            cmap=cmap
        )


# -----------------------------------------------------------------------------
# Functional gene-set scoring
# -----------------------------------------------------------------------------

def load_functional_gene_sets(gene_set_file, encoding="GBK"):
    """Load functional gene sets from CSV file."""

    gene_df = pd.read_csv(gene_set_file, encoding=encoding)
    gene_df = gene_df.dropna(how="all")

    column_map = {
        "Exhaustion_score": "Exhaustion scores",
        "Apoptosis_score": "Apoptosis scores",
        "IFN_score": "IFN_Scores",
        "Unhelp_score": "Unhelp_Scores"
    }

    gene_sets = {}

    for score_name, column_name in column_map.items():
        if column_name not in gene_df.columns:
            print(f"Warning: column not found in gene-set file: {column_name}")
            continue

        genes = [
            str(gene).strip()
            for gene in gene_df[column_name].dropna().unique()
            if str(gene).strip() != ""
        ]

        gene_sets[score_name] = genes

    return gene_sets


def score_gene_sets(adata_obj, gene_sets):
    """Calculate gene-set scores."""

    for score_name, genes in gene_sets.items():
        genes_present = [
            gene for gene in genes
            if gene in adata_obj.var_names
        ]

        print(
            f"Scoring {score_name}: "
            f"{len(genes_present)}/{len(genes)} genes found"
        )

        if len(genes_present) == 0:
            print(f"Skipped {score_name}: no genes found in adata.var_names.")
            continue

        sc.tl.score_genes(
            adata_obj,
            gene_list=genes_present,
            score_name=score_name,
            use_raw=False
        )

    return adata_obj


def plot_score_umaps(adata_cd4, gene_sets, outdir, cmap):
    """Plot UMAPs for functional scores."""

    for score_name in gene_sets.keys():
        if score_name not in adata_cd4.obs.columns:
            continue

        vmin, vmax = get_score_limits(adata_cd4, score_name)

        sc.pl.umap(
            adata_cd4,
            color=[score_name],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            size=45,
            show=False
        )

        save_figure(
            Path(outdir) / f"2.1.0_CD4T_{score_name}_UMAP.pdf"
        )


def plot_score_umaps_by_condition(
    adata_cd4,
    gene_sets,
    outdir,
    cmap,
    conditions=("HC", "AC", "CO")
):
    """Plot functional score UMAPs split by condition."""

    if "X_umap" not in adata_cd4.obsm:
        raise ValueError("X_umap was not found in adata_cd4.obsm.")

    umap_coords = adata_cd4.obsm["X_umap"]

    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()

    for score_name in gene_sets.keys():
        if score_name not in adata_cd4.obs.columns:
            continue

        print(f"Plotting {score_name} by condition")

        vmin, vmax = get_score_limits(adata_cd4, score_name)

        fig, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(5 * len(conditions), 4)
        )

        if len(conditions) == 1:
            axes = [axes]

        for j, condition in enumerate(conditions):
            adata_sub = adata_cd4[
                adata_cd4.obs["condition"] == condition
            ].copy()

            sc.pl.umap(
                adata_sub,
                color=score_name,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                size=80,
                ax=axes[j],
                show=False,
                title=f"{condition} - {score_name}"
            )

            axes[j].set_xlim(x_min, x_max)
            axes[j].set_ylim(y_min, y_max)
            axes[j].set_aspect("equal")

        plt.tight_layout()

        save_figure(
            Path(outdir) / f"2.1.1_CD4T_{score_name}_conditions.pdf"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="CD4+ T cell analysis for neonatal sepsis scRNA-seq data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input AnnData .h5ad file."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory."
    )

    parser.add_argument(
        "--gene-sets",
        default=None,
        help="CSV file containing functional gene sets."
    )

    parser.add_argument(
        "--gene-set-encoding",
        default="GBK",
        help="Encoding of the functional gene-set CSV file. Default: GBK."
    )

    parser.add_argument(
        "--color-map",
        default=None,
        help="Optional CSV file containing celltype-color mapping."
    )

    parser.add_argument(
        "--root-celltype",
        default="CD4_03_Naive_CCR7_TCF7",
        help="Root CD4+ T cell subtype for DPT pseudotime."
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["HC", "AC", "CO"],
        help="Condition order for plotting and comparisons."
    )

    parser.add_argument(
        "--skip-deg",
        action="store_true",
        help="Skip DEG analysis."
    )

    parser.add_argument(
        "--skip-go",
        action="store_true",
        help="Skip GO enrichment analysis."
    )

    parser.add_argument(
        "--organism",
        default="Human",
        help="Organism for Enrichr. Default: Human."
    )

    parser.add_argument(
        "--go-library",
        default="GO_Biological_Process_2025",
        help="GO library name for Enrichr."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    make_dir(args.outdir)

    bluegreen_cmap, green_purple_cmap, pink_purple_cmap = make_colormaps()
    celltype_colors = load_celltype_colors(args.color_map)

    _, adata_cd4 = load_cd4_data(args.input)

    # Marker dotplot
    plot_cd4_marker_dotplot(
        adata_cd4=adata_cd4,
        outdir=args.outdir,
        cmap=bluegreen_cmap
    )

    # DPT pseudotime and PAGA
    _, df_pseudotime = run_cd4_pseudotime(
        adata_cd4=adata_cd4,
        outdir=args.outdir,
        root_celltype=args.root_celltype
    )

    plot_pseudotime_ridge(
        df_pseudotime=df_pseudotime,
        outdir=args.outdir,
        color_map=celltype_colors
    )

    # DEG and GO enrichment
    deg_excel = None

    if not args.skip_deg:
        deg_excel = run_cd4_deg_analysis(
            adata_cd4=adata_cd4,
            outdir=args.outdir
        )

    if not args.skip_go:
        run_go_enrichment(
            excel_path=deg_excel,
            outdir=args.outdir,
            cmap=pink_purple_cmap,
            organism=args.organism,
            go_library=args.go_library
        )

    # Functional gene-set scoring
    if args.gene_sets is not None:
        gene_sets = load_functional_gene_sets(
            gene_set_file=args.gene_sets,
            encoding=args.gene_set_encoding
        )

        adata_cd4 = score_gene_sets(
            adata_obj=adata_cd4,
            gene_sets=gene_sets
        )

        plot_score_umaps(
            adata_cd4=adata_cd4,
            gene_sets=gene_sets,
            outdir=args.outdir,
            cmap=green_purple_cmap
        )

        plot_score_umaps_by_condition(
            adata_cd4=adata_cd4,
            gene_sets=gene_sets,
            outdir=args.outdir,
            cmap=green_purple_cmap,
            conditions=args.conditions
        )

    # Save processed object
    adata_cd4.write_h5ad(
        Path(args.outdir) / "cd4_t_cell_analysis_processed.h5ad"
    )

    print("CD4+ T cell analysis completed successfully.")


if __name__ == "__main__":
    main()
