## Code for PAGA, differential expression, and GO enrichment analyses using myeloid cells as an example

# Read annotated dataset
adata = sc.read_h5ad("annotated_data.h5ad")

# Myeloid cell subsets
adata_myeloid = adata [adata.obs['celltype_Major'].isin(['Mono','mDC','pDC','RBC','Mega']),].copy()

# Monocytes PAGA analysis

adata_mono_1 = adata_mono[
    (adata_mono.obs["celltype_Subset"] != "mDC") &
    (adata_mono.obs["celltype_Subset"] != "pDC") &
    (adata_mono.obs["celltype_Subset"] != "Mega") &
    (adata_mono.obs["celltype_Subset"] != "RBC")
].copy()

sc.pp.neighbors(
    adata_mono_1,
    use_rep="X_pca_harmony",
    n_pcs=20,
    n_neighbors=30,
    metric="cosine"
)

sc.tl.diffmap(adata_mono_1)

dc1 = adata_mono_1.obsm["X_diffmap"][:, 0]
iroot = int(np.argmin(dc1))
adata_mono_1.uns["iroot"] = iroot

sc.tl.dpt(
    adata_mono_1,
    n_dcs=10,
    n_branchings=0,
    allow_kendall_tau_shift=False
)

sc.tl.paga(adata_mono_1, groups="celltype_Subset")

sc.pl.paga(
    adata_mono_1,
    node_size_scale=1.5,
    node_size_power=0.2,
    edge_width_scale=1,
    threshold=0.03,
    color=["dpt_pseudotime"],
    show=False
)

plt.tight_layout()
plt.savefig("paga_mono_time.pdf")
plt.show()

# Differential expression analysis

def deg_analysis(adata, group, ref, p_adj_thresh=0.05, logfc_thresh=0.5):
    sc.tl.rank_genes_groups(
        adata,
        groupby="condition",
        groups=[group],
        reference=ref,
        method="wilcoxon",
        use_raw=False
    )

    df = sc.get.rank_genes_groups_df(adata, group=group)
    df_sig = df[
        (df["pvals_adj"] < p_adj_thresh) &
        (df["logfoldchanges"] > logfc_thresh)
    ]

    return df, df_sig


adata_mono_MDSC = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("MDSC")
].copy()

adata_mono_Classical = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("Classical")
].copy()

adata_mono_Activated = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("Activated")
].copy()

adata_mono_Intermediate = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("Intermediate")
].copy()

adata_mono_Stress = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("Stress")
].copy()

adata_mono_Dividing = adata_mono[
    adata_mono.obs["celltype_Subset"].str.contains("Dividing")
].copy()


datasets = {
    "mono_MDSC": adata_mono_MDSC,
    "mono_Classical": adata_mono_Classical,
    "mono_Activated": adata_mono_Activated,
    "mono_Intermediate": adata_mono_Intermediate,
}

comparisons = [
    ("AC", "HC"),
    ("CO", "HC"),
    ("CO", "AC"),
]

deg_results_all = {}
sig_deg_results_all = {}

for dataset_name, adata_obj in datasets.items():
    for grp, ref in comparisons:
        comp_name = f"{dataset_name}_{grp}_vs_{ref}"

        df_all, df_sig = deg_analysis(adata_obj, grp, ref)

        deg_results_all[comp_name] = df_all
        sig_deg_results_all[comp_name] = df_sig

        print(f"{comp_name}: significant DEGs = {len(df_sig)}")


excel_sig = "mono_condition_DEG_sig.xlsx"

with pd.ExcelWriter(excel_sig, mode="w", engine="openpyxl") as writer:
    for comp_name, df_sig in sig_deg_results_all.items():
        sheet_name = comp_name if len(comp_name) <= 31 else comp_name[:31]
        df_sig.to_excel(writer, sheet_name=sheet_name, index=False)


excel_all = "mono_condition_DEG_all.xlsx"

with pd.ExcelWriter(excel_all, mode="w", engine="openpyxl") as writer:
    for comp_name, df_all in deg_results_all.items():
        sheet_name = comp_name if len(comp_name) <= 31 else comp_name[:31]
        df_all.to_excel(writer, sheet_name=sheet_name, index=False)

print("Differential expression analysis completed.")

# GO enrichment analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from matplotlib.colors import LinearSegmentedColormap

excel_path = "mono_condition_DEG_sig.xlsx"
xls = pd.ExcelFile(excel_path)
sheet_names = xls.sheet_names

go_sets = ["GO_Biological_Process_2025"]

outdir_base = "GO_results"
os.makedirs(outdir_base, exist_ok=True)


def plot_bar(go_df, topn=20, title=None, savepath=None):
    df_plot = go_df.sort_values("Adjusted P-value").head(topn).copy()
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
        legend=False
    )

    plt.xlim(0, df_plot["log10P"].max())
    plt.xlabel(r"$-\log_{10}$(P-value)", fontsize=12)
    plt.ylabel("")
    plt.title(title, fontsize=14)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


for sheet in sheet_names:
    print(f"Running GO enrichment: {sheet}")

    df_sig_comp = pd.read_excel(excel_path, sheet_name=sheet)
    genes = df_sig_comp["names"].dropna().unique().tolist()

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=go_sets,
        organism="Human",
        outdir=None,
        cutoff=1
    )

    go_df = enr.results

    save_dir = os.path.join(outdir_base, sheet)
    os.makedirs(save_dir, exist_ok=True)

    go_df.to_csv(os.path.join(save_dir, f"{sheet}.csv"), index=False)

    plot_bar(
        go_df,
        topn=20,
        title=f"{sheet}_GO Enrichment",
        savepath=os.path.join(save_dir, f"{sheet}.pdf")
    )
