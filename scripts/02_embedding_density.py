"""
This script computes and visualizes cell density in UMAP embedding space
for each disease condition.

Input:
    Annotated AnnData object in .h5ad format

Output:
    UMAP density plots for each condition
"""

from pathlib import Path

import scanpy as sc
import matplotlib.pyplot as plt


def plot_embedding_density(
    input_h5ad,
    output_dir,
    groupby="condition",
    basis="umap",
):
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_h5ad)

    sc.tl.embedding_density(
        adata,
        basis=basis,
        groupby=groupby,
    )

    density_key = f"{basis}_density_{groupby}"

    groups = list(adata.obs[groupby].cat.categories)

    for group in groups:
        sc.pl.embedding_density(
            adata,
            basis=basis,
            key=density_key,
            group=group,
            show=False,
        )

        plt.savefig(
            output_dir / f"{basis}_density_{group}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    adata.write(output_dir / "adata_with_embedding_density.h5ad")

    return adata


if __name__ == "__main__":
    plot_embedding_density(
        input_h5ad="annotated_cells.h5ad",
        output_dir="embedding_density_results",
        groupby="condition",
        basis="umap",
    )
