#!/usr/bin/env python

"""
CD8+ T cell subcluster analysis for neonatal sepsis scRNA-seq data.

This script performs:
1. CD8+ T cell marker dotplot
2. Diffusion map, DPT pseudotime and PAGA analysis
3. Functional gene-set scoring
4. UMAP visualization of functional scores
5. Score comparison across conditions and CD8+ T cell subtypes
6. Differential expression analysis across conditions
7. GO enrichment analysis for significant DEGs

Example
-------
python scripts/06_cd8_t_cell_analysis.py \
    --input data/adata_10_celltype_Subset_log.h5ad \
    --gene_sets data/exhaustion_cytotoxic_IFN_genes.csv \
    --color_map data/celltype_color_map.csv \
    --outdir results/CD8T
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde, mannwhitneyu
from statsmodels.stats.multitest import multipletests


np.random.seed(0)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def make_dir(path):
    """Create output directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(path, dpi=300):
    """Save current matplotlib figure."""
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def make_colormaps():
    """Generate custom colormaps used in the analysis."""
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
    """Load cell-type color map from CSV file."""
    if color_map_file is None or not Path(color_map_file).exists():
        return {}

    color_df = pd.read_csv(color_map_file)
    if not {"celltype", "color"}.issubset(color_df.columns):
        raise ValueError("Color map file must contain 'celltype' and 'color' columns.")

    return dict(zip(color_df["celltype"], color_df["color"]))


# -----------------------------------------------------------------------------
# 1. Load data and subset CD8+ T cells
# -----------------------------------------------------------------------------

def load_cd8_data(input_file):
    """Load AnnData and subset CD8+ T cells."""
    adata = sc.read_h5ad(input_file)

    adata_cd8 = adata[
        adata.obs["celltype_Major"].isin(["CD8T"])
    ].copy()

    print(f"Loaded AnnData: {adata.shape}")
    print(f"CD8+ T cell subset: {adata_cd8.shape}")

    return adata, adata_cd8


# -----------------------------------------------------------------------------
# 2. Marker dotplot
# -----------------------------------------------------------------------------

def plot_cd8_marker_dotplot(adata_cd8, outdir, cmap):
    """Plot marker gene dotplot for CD8+ T cell subtypes."""

    marker_genes = [
        "CD3E", "CD8A", "CD8B",
        "SELL", "BACH2",
        "RUNX1", "ITM2A", "GZMK", "FOS", "CCL5",
        "S100B", "KLRK1", "KLRC4", "PRKCQ", "SPINK2",
        "ISG15", "MX1",
        "MKI67", "TYMS",
        "PRF1", "GZMA", "GZMB", "GNLY", "NKG7"
    ]

    marker_genes = [g for g in marker_genes if g in adata_cd8.var_names]

    sc.pl.dotplot(
        adata_cd8,
        var_names=marker_genes,
        groupby="celltype_Subset",
        standard_scale="var",
        dot_max=0.8,
        color_map=cmap,
        show=False
    )

    plt.gcf().set_size_inches(8, 4.5)
    save_figure(Path(outdir) / "1.1.3_cd8_cell_subtypes_dotplot.pdf")


# -----------------------------------------------------------------------------
# 3. Diffusion map, DPT pseudotime and PAGA
# -----------------------------------------------------------------------------

def run_cd8_pseudotime(adata_cd8, outdir):
    """Run diffusion map, DPT pseudotime and PAGA analysis."""

    adata_dpt = adata_cd8[
        adata_cd8.obs["celltype_Subset"] != "γδT"
    ].copy()

    if "X_pca_harmony" not in adata_dpt.obsm:
        raise ValueError(
            "X_pca_harmony not found in adata.obsm. "
            "Please run Harmony integration before DPT analysis."
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
        adata_dpt.obs["celltype_Subset"] == "CD8_01_Naive_SELL_BACH2"
    )

    if len(root_cells) == 0:
        raise ValueError("Root cell population CD8_01_Naive_SELL_BACH2 was not found.")

    adata_dpt.uns["iroot"] = root_cells[0]

    sc.tl.dpt(
        adata_dpt,
        n_dcs=10,
        n_branchings=1,
        allow_kendall_tau_shift=False
    )

    sc.tl.paga(adata_dpt, groups="celltype_Subset")

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
    save_figure(Path(outdir) / "1.3.1_paga_cd8_pseudotime.pdf")

    df_pseudotime = adata_dpt.obs[
        ["celltype_Subset", "dpt_pseudotime"]
    ].dropna().copy()

    df_pseudotime.to_csv(
        Path(outdir) / "1.3.1_cd8_pseudotime_values.csv",
        index=False
    )

    return adata_dpt, df_pseudotime


def plot_pseudotime_ridge(df_pseudotime, outdir, color_map=None):
    """Plot pseudotime density ridge plot using matplotlib."""

    order_df = (
        df_pseudotime
        .groupby("celltype_Subset")["dpt_pseudotime"]
        .median()
        .sort_values()
    )

    celltypes = list(order_df.index)
    y_positions = np.arange(len(celltypes))

    fig, ax = plt.subplots(figsize=(7, 5))

    x_grid = np.linspace(0, 1, 300)

    for i, celltype in enumerate(celltypes):
        values = df_pseudotime.loc[
            df_pseudotime["celltype_Subset"] == celltype,
            "dpt_pseudotime"
        ].dropna().values

        if len(values) < 5:
            continue

        kde = gaussian_kde(values)
        density = kde(x_grid)
        density = density / density.max() * 0.8

        color = color_map.get(celltype, "#96C0FF") if color_map else "#96C0FF"

        ax.fill_between(
            x_grid,
            i,
            i + density,
            alpha=0.7,
            color=color,
            linewidth=0.3
        )

        ax.plot(
            x_grid,
            i + density,
            color="black",
            linewidth=0.3
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(celltypes)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("CD8+ T cell pseudotime distribution")

    plt.tight_layout()
    save_figure(Path(outdir) / "1.3.2_cd8_pseudotime_ridgeplot.pdf")


# -----------------------------------------------------------------------------
# 4. Gene-set scoring
# -----------------------------------------------------------------------------

def load_gene_sets(gene_set_file):
    """Load functional gene sets from CSV file."""

    gene_df = pd.read_csv(gene_set_file, encoding="GBK")
    gene_df = gene_df.dropna(how="all")

    gene_sets = {
        "Exhaustion_score": "Exhaustion scores",
        "Cytotoxic_score": "Cytotoxic scores",
        "IFNG_score": "IFN-γ response score",
        "Regulatory_effector_score": "Regulatory effector score",
        "IFN_score": "IFN_Scores",
        "Unhelp_score": "Unhelp_Scores",
        "Apoptosis_score": "Apoptosis scores"
    }

    parsed_gene_sets = {}

    for score_name, col_name in gene_sets.items():
        if col_name not in gene_df.columns:
            print(f"Warning: column '{col_name}' not found in gene-set file.")
            continue

        genes = gene_df[col_name].dropna().unique().tolist()
        parsed_gene_sets[score_name] = genes

    return parsed_gene_sets


def score_gene_sets(adata_cd8, gene_sets):
    """Calculate gene-set scores using scanpy.tl.score_genes."""

    for score_name, genes in gene_sets.items():
        genes_present = [g for g in genes if g in adata_cd8.var_names]

        print(
            f"Scoring {score_name}: "
            f"{len(genes_present)}/{len(genes)} genes found"
        )

        if len(genes_present) == 0:
            print(f"Skipped {score_name}: no genes found in adata.var_names.")
            continue

        sc.tl.score_genes(
            adata_cd8,
            gene_list=genes_present,
            score_name=score_name,
            use_raw=False
        )

    return adata_cd8


# -----------------------------------------------------------------------------
# 5. UMAP visualization of scores
# -----------------------------------------------------------------------------

def plot_score_umaps(adata_cd8, gene_sets, outdir, cmap):
    """Plot UMAPs for each functional score."""

    for score_name in gene_sets.keys():
        if score_name not in adata_cd8.obs.columns:
            continue

        sc.pl.umap(
            adata_cd8,
            color=[score_name],
            cmap=cmap,
            size=45,
            show=False
        )

        save_figure(Path(outdir) / f"1.5.1_cd8_{score_name}_umap.pdf")


def plot_score_umaps_by_condition(
    adata_cd8,
    gene_sets,
    outdir,
    cmap,
    conditions=("HC", "AC", "CO")
):
    """Plot score UMAPs split by condition."""

    if "X_umap" not in adata_cd8.obsm:
        raise ValueError("X_umap not found in adata_cd8.obsm.")

    umap_coords = adata_cd8.obsm["X_umap"]
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()

    for score_name in gene_sets.keys():
        if score_name not in adata_cd8.obs.columns:
            continue

        print(f"Plotting {score_name} by condition")

        fig, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(5 * len(conditions), 4)
        )

        if len(conditions) == 1:
            axes = [axes]

        for j, cond in enumerate(conditions):
            ad_sub = adata_cd8[
                adata_cd8.obs["condition"] == cond,
                :
            ].copy()

            sc.pl.umap(
                ad_sub,
                color=score_name,
                cmap=cmap,
                size=80,
                ax=axes[j],
                show=False,
                title=f"{cond} - {score_name}"
            )

            axes[j].set_xlim(x_min, x_max)
            axes[j].set_ylim(y_min, y_max)
            axes[j].set_aspect("equal")

        plt.tight_layout()

        save_figure(
            Path(outdir) / f"1.5.2_cd8_{score_name}_conditions.pdf"
        )


# -----------------------------------------------------------------------------
# 6. Score boxplots and statistics
# -----------------------------------------------------------------------------

def export_score_dataframe(adata_cd8, score_names, outdir):
    """Export score dataframe for downstream plotting/statistics."""

    cols = ["celltype_Subset", "condition"] + score_names
    cols = [c for c in cols if c in adata_cd8.obs.columns]

    df = adata_cd8.obs[cols].copy()
    df = df[df["celltype_Subset"].str.contains("CD8_", na=False)]

    df.to_csv(Path(outdir) / "1.6_cd8_functional_scores.csv", index=False)

    return df


def pairwise_wilcoxon_by_celltype(df, score_name):
    """Pairwise Wilcoxon test between conditions within each cell type."""

    results = []

    for celltype, sub_df in df.groupby("celltype_Subset"):
        conditions = sub_df["condition"].dropna().unique().tolist()

        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]

                x = sub_df.loc[sub_df["condition"] == cond1, score_name].dropna()
                y = sub_df.loc[sub_df["condition"] == cond2, score_name].dropna()

                if len(x) < 3 or len(y) < 3:
                    continue

                stat, pval = mannwhitneyu(x, y, alternative="two-sided")

                results.append({
                    "celltype_Subset": celltype,
                    "score": score_name,
                    "group1": cond1,
                    "group2": cond2,
                    "p_value": pval
                })

    stat_df = pd.DataFrame(results)

    if not stat_df.empty:
        stat_df["p_adj"] = multipletests(
            stat_df["p_value"],
            method="fdr_bh"
        )[1]

    return stat_df


def plot_score_boxplots(df, score_names, outdir):
    """Plot score boxplots grouped by condition and cell subtype."""

    condition_colors = {
        "HC": "#c6cdf7",
        "AC": "#e6a0c4",
        "CO": "#f2c396"
    }

    all_stats = []

    for score_name in score_names:
        if score_name not in df.columns:
            continue

        print(f"Plotting boxplot for {score_name}")

        plt.figure(figsize=(8, 6))

        sns.boxplot(
            data=df,
            x="celltype_Subset",
            y=score_name,
            hue="condition",
            palette=condition_colors,
            width=0.7,
            showfliers=False
        )

        plt.title(score_name)
        plt.xlabel("")
        plt.ylabel("Score")
        plt.xticks(rotation=90)
        plt.legend(title="condition", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        save_figure(
            Path(outdir) / f"1.6.1_CD8T_{score_name}_boxplot_condition.pdf"
        )

        stat_df = pairwise_wilcoxon_by_celltype(df, score_name)
        if not stat_df.empty:
            all_stats.append(stat_df)

        plt.figure(figsize=(6, 6))

        sns.boxplot(
            data=df,
            x="celltype_Subset",
            y=score_name,
            color="#96C0FF",
            width=0.6,
            showfliers=False
        )

        plt.title(score_name)
        plt.xlabel("")
        plt.ylabel("Score")
        plt.xticks(rotation=90)
        plt.tight_layout()

        save_figure(
            Path(outdir) / f"1.6.2_CD8T_{score_name}_boxplot_celltypes.pdf"
        )

    if all_stats:
        all_stats_df = pd.concat(all_stats, axis=0)
        all_stats_df.to_csv(
            Path(outdir) / "1.6_CD8T_score_wilcoxon_statistics.csv",
            index=False
        )


# -----------------------------------------------------------------------------
# 7. Differential expression analysis
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
        print(f"Skipped {group} vs {ref}: missing group.")
        return None, None

    if counts[group] < min_cells or counts[ref] < min_cells:
        print(f"Skipped {group} vs {ref}: too few cells.")
        return None, None

    sc.tl.rank_genes_groups(
        adata_obj,
        groupby=groupby,
        groups=[group],
        reference=ref,
        method="wilcoxon",
        use_raw=False
    )

    df_all = sc.get.rank_genes_groups_df(adata_obj, group=group)

    df_sig = df_all[
        (df_all["pvals_adj"] < p_adj_thresh) &
        (df_all["logfoldchanges"] > logfc_thresh)
    ].copy()

    return df_all, df_sig


def run_cd8_deg_analysis(adata_cd8, outdir):
    """Run DEG analysis for CD8+ T cell subtypes across conditions."""

    datasets = {
        "naive": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "Naive", case=False, na=False
            )
        ].copy(),

        "Memory": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "Memory_RUNX1", case=False, na=False
            )
        ].copy(),

        "eMemory": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "eMemory", case=False, na=False
            )
        ].copy(),

        "effector": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "KLR", case=False, na=False
            )
        ].copy(),

        "Dividing": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "Dividing", case=False, na=False
            )
        ].copy(),

        "effector_MX1": adata_cd8[
            adata_cd8.obs["celltype_Subset"].str.contains(
                "Effector_ISG15_MX1", case=False, na=False
            )
        ].copy()
    }

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
                adata_obj,
                group=group,
                ref=ref
            )

            if df_all is None:
                continue

            deg_results_all[comp_name] = df_all
            deg_results_sig[comp_name] = df_sig

            print(f"{comp_name}: {len(df_sig)} significant DEGs")

    excel_sig = Path(outdir) / "1.8_CD8T_condition_DEG_sig.xlsx"
    excel_all = Path(outdir) / "1.8_CD8T_condition_DEG_all.xlsx"

    with pd.ExcelWriter(excel_sig, mode="w", engine="openpyxl") as writer:
        for comp_name, df_sig in deg_results_sig.items():
            sheet_name = comp_name[:31]
            df_sig.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(excel_all, mode="w", engine="openpyxl") as writer:
        for comp_name, df_all in deg_results_all.items():
            sheet_name = comp_name[:31]
            df_all.to_excel(writer, sheet_name=sheet_name, index=False)

    print("DEG analysis finished.")

    return excel_sig


# -----------------------------------------------------------------------------
# 8. GO enrichment analysis
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

    norm = plt.Normalize(df_plot["log10P"].min(), df_plot["log10P"].max())
    df_plot["color"] = [cmap(norm(v)) for v in df_plot["log10P"]]

    plt.figure(figsize=(6, 5))

    sns.barplot(
        data=df_plot,
        x="log10P",
        y="Term",
        hue="Term",
        palette=df_plot.set_index("Term")["color"].to_dict(),
        legend=False
    )

    plt.xlabel(r"$-\log_{10}$(adjusted P-value)")
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()

    save_figure(savepath)


def run_go_enrichment(excel_path, outdir, cmap):
    """Run GO enrichment analysis for significant DEGs."""

    go_outdir = Path(outdir) / "1.8_GO_results"
    make_dir(go_outdir)

    sheet_names = pd.ExcelFile(excel_path).sheet_names
    go_sets = ["GO_Biological_Process_2025"]

    for sheet in sheet_names:
        print(f"Running GO enrichment for: {sheet}")

        df_sig = pd.read_excel(excel_path, sheet_name=sheet)

        if "names" not in df_sig.columns:
            print(f"Skipped {sheet}: no 'names' column.")
            continue

        genes = df_sig["names"].dropna().unique().tolist()

        if len(genes) < 5:
            print(f"Skipped {sheet}: too few genes.")
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=go_sets,
                organism="Human",
                outdir=None,
                cutoff=1
            )
        except Exception as e:
            print(f"GO enrichment failed for {sheet}: {e}")
            continue

        go_df = enr.results

        save_dir = go_outdir / sheet
        make_dir(save_dir)

        go_df.to_csv(save_dir / f"{sheet}_GO_enrichment.csv", index=False)

        plot_go_barplot(
            go_df,
            topn=20,
            title=f"{sheet} GO enrichment",
            savepath=save_dir / f"{sheet}_GO_enrichment.pdf",
            cmap=cmap
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CD8+ T cell analysis for neonatal sepsis scRNA-seq data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input AnnData h5ad file."
    )

    parser.add_argument(
        "--gene_sets",
        required=True,
        help="CSV file containing functional gene sets."
    )

    parser.add_argument(
        "--color_map",
        default=None,
        help="Optional CSV file containing celltype-color mapping."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    make_dir(args.outdir)

    bluegreen_cmap, green_purple_cmap, pink_purple_cmap = make_colormaps()
    celltype_colors = load_celltype_colors(args.color_map)

    adata, adata_cd8 = load_cd8_data(args.input)

    # Marker dotplot
    plot_cd8_marker_dotplot(
        adata_cd8=adata_cd8,
        outdir=args.outdir,
        cmap=bluegreen_cmap
    )

    # Pseudotime and PAGA
    adata_dpt, df_pseudotime = run_cd8_pseudotime(
        adata_cd8=adata_cd8,
        outdir=args.outdir
    )

    plot_pseudotime_ridge(
        df_pseudotime=df_pseudotime,
        outdir=args.outdir,
        color_map=celltype_colors
    )

    # Gene-set scoring
    gene_sets = load_gene_sets(args.gene_sets)

    adata_cd8 = score_gene_sets(
        adata_cd8=adata_cd8,
        gene_sets=gene_sets
    )

    score_names = list(gene_sets.keys())

    # UMAP visualization
    plot_score_umaps(
        adata_cd8=adata_cd8,
        gene_sets=gene_sets,
        outdir=args.outdir,
        cmap=green_purple_cmap
    )

    plot_score_umaps_by_condition(
        adata_cd8=adata_cd8,
        gene_sets=gene_sets,
        outdir=args.outdir,
        cmap=green_purple_cmap
    )

    # Boxplots and statistics
    score_df = export_score_dataframe(
        adata_cd8=adata_cd8,
        score_names=score_names,
        outdir=args.outdir
    )

    plot_score_boxplots(
        df=score_df,
        score_names=score_names,
        outdir=args.outdir
    )

    # DEG and GO enrichment
    deg_excel = run_cd8_deg_analysis(
        adata_cd8=adata_cd8,
        outdir=args.outdir
    )

    run_go_enrichment(
        excel_path=deg_excel,
        outdir=args.outdir,
        cmap=pink_purple_cmap
    )

    # Save processed CD8 object with scores
    adata_cd8.write_h5ad(Path(args.outdir) / "cd8_t_cell_analysis_with_scores.h5ad")

    print("CD8+ T cell analysis completed successfully.")


if __name__ == "__main__":
    main()
