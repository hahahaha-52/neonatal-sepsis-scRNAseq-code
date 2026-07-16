"""
This script performs CellPhoneDB analysis using annotated single-cell RNA-seq data.

Analyses include:
1. Preparing condition-specific AnnData files and metadata files
2. Running CellPhoneDB statistical analysis for HC, AC, and CO groups
3. Plotting CellPhoneDB interaction heatmaps for each group
4. Plotting Treg-associated ligand-receptor interactions for each group

Input:
    Annotated AnnData object in .h5ad format

Output:
    Condition-specific .h5ad files, metadata files, CellPhoneDB results,
    heatmaps, and Treg-associated interaction plots
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import ktplotspy as kpy

np.random.seed(0)


def prepare_condition_h5ad_files(
    input_h5ad,
    output_dir,
    condition_key="condition",
    celltype_key="celltype_Subset",
    conditions=("HC", "AC", "CO"),
    use_raw=True,
):
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    if use_raw and adata.raw is not None:
        adata_expr = adata.raw.to_adata()
        adata_expr.obs = adata.obs.copy()
    else:
        adata_expr = adata.copy()

    output_files = {}

    for condition in conditions:
        adata_condition = adata_expr[
            adata_expr.obs[condition_key].astype(str) == condition
        ].copy()

        h5ad_path = output_dir / f"adata_{condition}.h5ad"
        meta_path = output_dir / f"meta_{condition}.csv"

        adata_condition.write(h5ad_path)

        meta = pd.DataFrame(
            {
                "Cell": adata_condition.obs_names,
                "cell_type": adata_condition.obs[celltype_key].astype(str).values,
            }
        )

        meta.to_csv(meta_path, index=False)

        output_files[condition] = {
            "h5ad": h5ad_path,
            "meta": meta_path,
        }

        print(f"{condition}: saved {adata_condition.n_obs} cells")

    return output_files


def run_cellphonedb_for_conditions(
    input_files,
    output_dir,
    cpdb_file_path=None,
    counts_data="hgnc_symbol",
    iterations=1000,
    threshold=0.1,
    result_precision=3,
    threads=8,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for condition, files in input_files.items():
        condition_output = output_dir / f"out_{condition}"
        condition_output.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "meta_file_path": str(files["meta"]),
            "counts_file_path": str(files["h5ad"]),
            "counts_data": counts_data,
            "output_path": str(condition_output),
            "iterations": iterations,
            "threshold": threshold,
            "result_precision": result_precision,
            "threads": threads,
        }

        if cpdb_file_path is not None:
            kwargs["cpdb_file_path"] = str(cpdb_file_path)

        print(f"Running CellPhoneDB for {condition}")

        results[condition] = cpdb_statistical_analysis_method.call(**kwargs)

    return results


def find_cpdb_result_files(condition_output_dir):
    condition_output_dir = Path(condition_output_dir)

    mean_files = sorted(condition_output_dir.glob("statistical_analysis_means*.txt"))
    pvalue_files = sorted(condition_output_dir.glob("statistical_analysis_pvalues*.txt"))

    if len(mean_files) == 0:
        raise FileNotFoundError(
            f"No CellPhoneDB mean file found in {condition_output_dir}"
        )

    if len(pvalue_files) == 0:
        raise FileNotFoundError(
            f"No CellPhoneDB p-value file found in {condition_output_dir}"
        )

    return mean_files[-1], pvalue_files[-1]


def save_plot(plot_object, output_path):
    output_path = Path(output_path)

    if hasattr(plot_object, "savefig"):
        plot_object.savefig(output_path)

    elif hasattr(plot_object, "save"):
        plot_object.save(str(output_path))

    else:
        raise TypeError("Unsupported plot object type.")


def plot_cpdb_heatmaps(
    cpdb_output_dir,
    conditions=("HC", "AC", "CO"),
    figsize=(25, 25),
):
    cpdb_output_dir = Path(cpdb_output_dir)

    for condition in conditions:
        condition_output = cpdb_output_dir / f"out_{condition}"
        _, pvals_path = find_cpdb_result_files(condition_output)

        pvals = pd.read_csv(pvals_path, sep="\t")

        plot_obj = kpy.plot_cpdb_heatmap(
            pvals=pvals,
            figsize=figsize,
            title=f"{condition}: sum of significant interactions",
            symmetrical=False,
        )

        save_plot(
            plot_obj,
            condition_output / f"cellphonedb_heatmap_{condition}.pdf",
        )

        print(f"Saved heatmap for {condition}")


def plot_selected_cpdb_interactions(
    adata_path,
    cpdb_output_dir,
    condition,
    cell_type1="CD4_09_Treg_SELL_FOXP3|CD4_10_Treg_FOXP3_IL2RA",
    cell_type2=".",
    celltype_key="celltype_Subset",
    figsize=(25, 25),
):
    adata = sc.read_h5ad(adata_path)

    available_celltypes = set(adata.obs[celltype_key].astype(str).unique())
    query_celltypes = cell_type1.split("|")

    matched_celltypes = [
        celltype for celltype in query_celltypes
        if celltype in available_celltypes
    ]

    if len(matched_celltypes) == 0:
        print(
            f"{condition}: skipped Treg interaction plot because none of the "
            f"specified Treg subsets were found."
        )
        return

    cell_type1_filtered = "|".join(matched_celltypes)

    condition_output = Path(cpdb_output_dir) / f"out_{condition}"
    means_path, pvals_path = find_cpdb_result_files(condition_output)

    means = pd.read_csv(means_path, sep="\t")
    pvals = pd.read_csv(pvals_path, sep="\t")

    plot_obj = kpy.plot_cpdb(
        adata=adata,
        cell_type1=cell_type1_filtered,
        cell_type2=cell_type2,
        means=means,
        pvals=pvals,
        celltype_key=celltype_key,
        figsize=figsize,
        default_style=False,
        title=f"{condition}: Treg-associated CellPhoneDB interactions",
    )

    save_plot(
        plot_obj,
        condition_output / f"treg_interactions_{condition}.pdf",
    )

    print(f"Saved Treg-associated interaction plot for {condition}")


def plot_treg_interactions_for_conditions(
    prepared_input_dir,
    cpdb_output_dir,
    conditions=("HC", "AC", "CO"),
    cell_type1="CD4_09_Treg_SELL_FOXP3|CD4_10_Treg_FOXP3_IL2RA",
    cell_type2=".",
    celltype_key="celltype_Subset",
    figsize=(25, 25),
):
    prepared_input_dir = Path(prepared_input_dir)

    for condition in conditions:
        adata_path = prepared_input_dir / f"adata_{condition}.h5ad"

        if not adata_path.exists():
            print(f"{condition}: skipped because {adata_path} was not found.")
            continue

        plot_selected_cpdb_interactions(
            adata_path=adata_path,
            cpdb_output_dir=cpdb_output_dir,
            condition=condition,
            cell_type1=cell_type1,
            cell_type2=cell_type2,
            celltype_key=celltype_key,
            figsize=figsize,
        )


def main():
    input_h5ad = "annotated_cells.h5ad"
    prepared_input_dir = Path("cellphonedb_inputs")
    cpdb_output_dir = Path("cellphonedb_results")

    conditions = ("HC", "AC", "CO")

    input_files = prepare_condition_h5ad_files(
        input_h5ad=input_h5ad,
        output_dir=prepared_input_dir,
        condition_key="condition",
        celltype_key="celltype_Subset",
        conditions=conditions,
        use_raw=True,
    )

    run_cellphonedb_for_conditions(
        input_files=input_files,
        output_dir=cpdb_output_dir,
        cpdb_file_path=None,
        counts_data="hgnc_symbol",
        iterations=1000,
        threshold=0.1,
        result_precision=3,
        threads=8,
    )

    plot_cpdb_heatmaps(
        cpdb_output_dir=cpdb_output_dir,
        conditions=conditions,
        figsize=(25, 25),
    )

    plot_treg_interactions_for_conditions(
        prepared_input_dir=prepared_input_dir,
        cpdb_output_dir=cpdb_output_dir,
        conditions=conditions,
        cell_type1="CD4_09_Treg_SELL_FOXP3|CD4_10_Treg_FOXP3_IL2RA",
        cell_type2=".",
        celltype_key="celltype_Subset",
        figsize=(25, 25),
    )


if __name__ == "__main__":
    main()
