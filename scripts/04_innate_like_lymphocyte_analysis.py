from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

np.random.seed(0)


SUBSET_ORDER = [
    "NK_01_Naive_SELL_IL2RB",
    "NK_02_CD16_FCGR3A_KIR2DL1",
    "NK_03_CD16_FCGR3A_PRF1",
    "NK_04_CD56_NCAM1_XCL1",
    "NK_05_Dividing_MKI67_TUBA1B",
    "NKT_01_CD8A_NKG7",
    "NKT_02_Exhausted_LAG3_PDCD1",
    "γδT",
]


def safe_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("γ", "gamma")
        .replace("δ", "delta")
        .replace("+", "pos")
    )


def extract_innate_like_lymphocytes(
    adata,
    major_key="celltype_Major",
    subset_key="celltype_Subset",
):
    adata_nk = adata[
        adata.obs[major_key].isin(["NK", "γδT"])
    ].copy()

    present_categories = [
        item for item in SUBSET_ORDER
        if item in adata_nk.obs[subset_key].astype(str).unique()
    ]

    adata_nk.obs[subset_key] = pd.Categorical(
        adata_nk.obs[subset_key],
        categories=present_categories,
        ordered=True,
    )

    return adata_nk


def run_paga_dpt(
    adata_nk,
    output_dir,
    subset_key="celltype_Subset",
    use_rep="X_pca_harmony",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_paga = adata_nk.copy()

    sc.pp.neighbors(
        adata_paga,
        use_rep=use_rep,
        n_pcs=20,
        n_neighbors=30,
        metric="cosine",
    )

    sc.tl.diffmap(adata_paga)

    dc1 = adata_paga.obsm["X_diffmap"][:, 0]
    iroot = int(np.argmin(dc1))
    adata_paga.uns["iroot"] = iroot

    sc.tl.dpt(
        adata_paga,
        n_dcs=10,
        n_branchings=0,
        allow_kendall_tau_shift=False,
    )

    sc.tl.paga(adata_paga, groups=subset_key)

    sc.pl.paga(
        adata_paga,
        node_size_scale=1.5,
        node_size_power=0.2,
        edge_width_scale=1,
        threshold=0.03,
        color=["dpt_pseudotime"],
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "paga_innate_like_lymphocytes_pseudotime.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    adata_paga.write(output_dir / "adata_innate_like_lymphocytes_paga_dpt.h5ad")

    return adata_paga


def plot_pseudotime_density(
    adata,
    output_dir,
    subset_key="celltype_Subset",
    pseudotime_key="dpt_pseudotime",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = adata.obs[[subset_key, pseudotime_key]].dropna().copy()
    df[subset_key] = df[subset_key].astype(str)

    order_df = (
        df.groupby(subset_key)[pseudotime_key]
        .median()
        .sort_values()
    )

    subset_order = list(order_df.index)
    grid = np.linspace(0, 1, 300)

    cmap = plt.get_cmap("tab20")
    color_dict = {
        subset: cmap(i % 20)
        for i, subset in enumerate(subset_order)
    }

    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(subset_order))))

    for i, subset in enumerate(reversed(subset_order)):
        values = df.loc[df[subset_key] == subset, pseudotime_key].values

        if len(values) < 2:
            continue

        kde = gaussian_kde(values)
        density = kde(grid)
        density = density / density.max() * 0.8

        ax.fill_between(
            grid,
            i,
            i + density,
            color=color_dict[subset],
            alpha=0.7,
        )

        ax.plot(
            grid,
            i + density,
            color="black",
            linewidth=0.4,
        )

    ax.set_yticks(range(len(subset_order)))
    ax.set_yticklabels(list(reversed(subset_order)))
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        output_dir / "innate_like_lymphocytes_pseudotime_density.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def deg_analysis(
    adata,
    group,
    ref,
    condition_key="condition",
    p_adj_thresh=0.05,
    logfc_thresh=0.5,
):
    sc.tl.rank_genes_groups(
        adata,
        groupby=condition_key,
        groups=[group],
        reference=ref,
        method="wilcoxon",
        use_raw=False,
    )

    df = sc.get.rank_genes_groups_df(adata, group=group)

    df_sig = df[
        (df["pvals_adj"] < p_adj_thresh)
        & (df["logfoldchanges"] > logfc_thresh)
    ].copy()

    return df, df_sig


def run_deg_analysis(
    adata_nk,
    output_dir,
    subset_key="celltype_Subset",
    condition_key="condition",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "nk_all": adata_nk,
        "nk_naive": adata_nk[
            adata_nk.obs[subset_key].str.contains("Naive", na=False)
        ].copy(),
        "nk_02_cd16": adata_nk[
            adata_nk.obs[subset_key].str.contains("02_CD16", na=False)
        ].copy(),
        "nk_03_cd16": adata_nk[
            adata_nk.obs[subset_key].str.contains("03_CD16", na=False)
        ].copy(),
        "nk_cd56": adata_nk[
            adata_nk.obs[subset_key].str.contains("04_CD56", na=False)
        ].copy(),
        "nk_dividing": adata_nk[
            adata_nk.obs[subset_key].str.contains("Dividing", na=False)
        ].copy(),
        "nkt_01": adata_nk[
            adata_nk.obs[subset_key].str.contains("NKT_01", na=False)
        ].copy(),
        "nkt_02": adata_nk[
            adata_nk.obs[subset_key].str.contains("NKT_02", na=False)
        ].copy(),
        "gamma_delta_t": adata_nk[
            adata_nk.obs[subset_key].str.contains("γδT", na=False)
        ].copy(),
    }

    comparisons = [
        ("AC", "HC"),
        ("CO", "HC"),
        ("CO", "AC"),
    ]

    deg_results_all = {}
    sig_deg_results_all = {}

    for dataset_name, adata_obj in datasets.items():
        if adata_obj.n_obs == 0:
            print(f"{dataset_name}: skipped because no cells were found.")
            continue

        available_conditions = set(
            adata_obj.obs[condition_key].dropna().astype(str)
        )

        for group, ref in comparisons:
            if group not in available_conditions or ref not in available_conditions:
                print(f"{dataset_name}_{group}_vs_{ref}: skipped because one group is missing.")
                continue

            comp_name = f"{dataset_name}_{group}_vs_{ref}"

            df_all, df_sig = deg_analysis(
                adata_obj,
                group=group,
                ref=ref,
                condition_key=condition_key,
            )

            deg_results_all[comp_name] = df_all
            sig_deg_results_all[comp_name] = df_sig

            print(f"{comp_name}: significant DEGs = {len(df_sig)}")

    excel_sig = output_dir / "innate_like_lymphocytes_condition_DEG_sig.xlsx"
    excel_all = output_dir / "innate_like_lymphocytes_condition_DEG_all.xlsx"

    with pd.ExcelWriter(excel_sig, mode="w", engine="openpyxl") as writer:
        for comp_name, df_sig in sig_deg_results_all.items():
            sheet_name = comp_name if len(comp_name) <= 31 else comp_name[:31]
            df_sig.to_excel(writer, sheet_name=sheet_name, index=False)

    with pd.ExcelWriter(excel_all, mode="w", engine="openpyxl") as writer:
        for comp_name, df_all in deg_results_all.items():
            sheet_name = comp_name if len(comp_name) <= 31 else comp_name[:31]
            df_all.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Differential expression analysis completed.")

    return excel_sig, excel_all


def plot_go_bar(go_df, topn=20, title=None, savepath=None):
    if go_df.empty:
        print(f"No GO terms found for {title}.")
        return

    df_plot = go_df.sort_values("Adjusted P-value").head(topn).copy()
    df_plot = df_plot[df_plot["Adjusted P-value"] > 0].copy()

    if df_plot.empty:
        print(f"No valid adjusted P-values found for {title}.")
        return

    df_plot["log10P"] = -np.log10(df_plot["Adjusted P-value"])

    colors = ["#f8d8e6", "#d18cd2", "#9a58c1"]
    cmap = LinearSegmentedColormap.from_list("pink_purple_gradient", colors)
    norm = plt.Normalize(df_plot["log10P"].min(), df_plot["log10P"].max())
    df_plot["color"] = [cmap(norm(v)) for v in df_plot["log10P"]]

    plt.figure(figsize=(6, 5))

    sns.barplot(
        data=df_plot,
        x="log10P",
        y="Term",
        hue="Term",
        palette=df_plot.set_index("Term")["color"].to_dict(),
        legend=False,
    )

    plt.xlim(0, df_plot["log10P"].max())
    plt.xlabel(r"$-\log_{10}$(adjusted P-value)", fontsize=12)
    plt.ylabel("")
    plt.title(title, fontsize=14)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.close()


def run_go_enrichment(
    deg_excel_path,
    output_dir,
    gene_sets=None,
    organism="Human",
):
    deg_excel_path = Path(deg_excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2025"]

    xls = pd.ExcelFile(deg_excel_path)

    for sheet in xls.sheet_names:
        print(f"Running GO enrichment: {sheet}")

        df_sig_comp = pd.read_excel(deg_excel_path, sheet_name=sheet)

        if "names" not in df_sig_comp.columns:
            print(f"{sheet}: skipped because the 'names' column was not found.")
            continue

        genes = df_sig_comp["names"].dropna().unique().tolist()

        if len(genes) == 0:
            print(f"{sheet}: skipped because no genes were found.")
            continue

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            cutoff=1,
        )

        go_df = enr.results

        save_dir = output_dir / safe_name(sheet)
        save_dir.mkdir(parents=True, exist_ok=True)

        go_df.to_csv(save_dir / f"{safe_name(sheet)}_GO_enrichment.csv", index=False)

        plot_go_bar(
            go_df,
            topn=20,
            title=f"{sheet} GO enrichment",
            savepath=save_dir / f"{safe_name(sheet)}_GO_enrichment.pdf",
        )

    print("GO enrichment analysis completed.")


def load_gene_sets(gene_signature_file):
    gene_df = pd.read_csv(gene_signature_file)
    gene_df = gene_df.dropna(how="all")

    gene_sets = {
        "Exhaustion_score": gene_df["Exhaustion scores"].dropna().unique().tolist(),
        "Cytotoxic_score": gene_df["Cytotoxic scores"].dropna().unique().tolist(),
        "IFNg_score": gene_df["IFN-γ response score"].dropna().unique().tolist(),
    }

    return gene_sets


def score_gene_sets(
    adata,
    gene_sets,
    use_raw=False,
):
    for score_name, genes in gene_sets.items():
        genes_present = [gene for gene in genes if gene in adata.var_names]

        if len(genes_present) == 0:
            print(f"{score_name}: skipped because no genes were found in the dataset.")
            continue

        print(f"Scoring {score_name}: {len(genes_present)} genes")

        sc.tl.score_genes(
            adata,
            gene_list=genes_present,
            score_name=score_name,
            use_raw=use_raw,
        )

    return adata


def plot_scores_by_condition(
    adata,
    output_dir,
    score_names,
    condition_key="condition",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list(
        "green_purple",
        ["#0E5D29", "#f7f7f7", "#40004C"],
    )

    conditions = [
        condition for condition in ["HC", "AC", "CO"]
        if condition in set(adata.obs[condition_key].astype(str))
    ]

    for score_name in score_names:
        if score_name not in adata.obs.columns:
            continue

        fig, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(5 * len(conditions), 4),
        )

        if len(conditions) == 1:
            axes = [axes]

        for i, condition in enumerate(conditions):
            adata_sub = adata[
                adata.obs[condition_key].astype(str) == condition
            ].copy()

            sc.pl.umap(
                adata_sub,
                color=score_name,
                cmap=cmap,
                size=80,
                ax=axes[i],
                show=False,
                title=f"{condition} - {score_name}",
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"innate_like_lymphocytes_{safe_name(score_name)}_conditions.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    input_h5ad = "annotated_cells.h5ad"
    gene_signature_file = "exhaustion_cytotoxic_IFN_genes.csv"
    output_dir = Path("innate_like_lymphocyte_analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    adata_nk = extract_innate_like_lymphocytes(
        adata,
        major_key="celltype_Major",
        subset_key="celltype_Subset",
    )

    adata_nk.write(output_dir / "adata_innate_like_lymphocytes.h5ad")

    adata_paga = run_paga_dpt(
        adata_nk,
        output_dir=output_dir / "PAGA",
        subset_key="celltype_Subset",
        use_rep="X_pca_harmony",
    )

    plot_pseudotime_density(
        adata_paga,
        output_dir=output_dir / "PAGA",
        subset_key="celltype_Subset",
        pseudotime_key="dpt_pseudotime",
    )

    deg_sig_path, deg_all_path = run_deg_analysis(
        adata_nk,
        output_dir=output_dir / "DEG",
        subset_key="celltype_Subset",
        condition_key="condition",
    )

    run_go_enrichment(
        deg_excel_path=deg_sig_path,
        output_dir=output_dir / "GO_results",
        gene_sets=["GO_Biological_Process_2025"],
        organism="Human",
    )

    gene_signature_path = Path(gene_signature_file)

    if gene_signature_path.exists():
        score_sets = load_gene_sets(gene_signature_path)
        adata_nk = score_gene_sets(
            adata_nk,
            gene_sets=score_sets,
            use_raw=False,
        )

        plot_scores_by_condition(
            adata_nk,
            output_dir=output_dir / "Score_UMAP",
            score_names=list(score_sets.keys()),
            condition_key="condition",
        )

        adata_nk.write(output_dir / "adata_innate_like_lymphocytes_scored.h5ad")
    else:
        print(f"{gene_signature_file} was not found. Gene set scoring was skipped.")


if __name__ == "__main__":
    main()
