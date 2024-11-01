import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional
from .constants import AminoAcid1Index

# Define the custom colors
LIGHT_BLUE = "#aec7e8"
BLUE = "#1f77b4"
DARK_BLUE = "#4c78a8"
SALMON_RED = "#d62728"

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_plddt(plddt: np.ndarray, lddt: np.ndarray, save_path: str = None):
    """
    Plot LDDT against Average pLDDT on the resolved region.

    :param lddt: LDDT scores of shape (N_batch,).
    :param plddt: pLDDT scores of shape (N_batch,).
    :param save_path: Path to save the figure. Defaults to None.
    """

    lddt = np.array(lddt)
    plddt = np.array(plddt)

    plt.figure(figsize=(8, 6))
    plt.scatter(plddt, lddt, s=50, alpha=0.6, color=DARK_BLUE, edgecolor="black")

    plt.plot([0, 100], [0, 100], linestyle="-", color="gray")
    plt.xlabel("pLDDT", fontsize=25)
    plt.ylabel("LDDT", fontsize=25)
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_sequence_distribution(
    pred_sequence: np.ndarray,
    true_sequence: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Create a bar plot to compare the distribution of amino acids
    in the predicted and true sequences.

    :param pred_sequence: Predicted amino acid indices with shape (n).
    :param true_sequence: True amino acid indices with shape (n).
    :param save_path: Optional path to save the plot. If None, the plot will be displayed.
    """

    pred_amino_acids = [AminoAcid1Index[aa] for aa in pred_sequence]
    true_amino_acids = [AminoAcid1Index[aa] for aa in true_sequence]
    data = pd.DataFrame(
        {
            "Amino Acid": pred_amino_acids + true_amino_acids,
            "Type": ["Generated"] * len(pred_amino_acids)
            + ["True"] * len(true_amino_acids),
        }
    )

    amino_acid_counts = (
        data.groupby(["Amino Acid", "Type"]).size().reset_index(name="Count")
    )

    total_counts = data["Type"].value_counts()
    amino_acid_counts["Density"] = amino_acid_counts.apply(
        lambda row: row["Count"] / total_counts[row["Type"]], axis=1
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Amino Acid",
        y="Density",
        hue="Type",
        data=amino_acid_counts,
        width=0.8,
        alpha=0.8,
        palette={"Generated": BLUE, "True": SALMON_RED},
        edgecolor="black",
    )

    plt.xlabel("Amino Acid", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.title("Amino Acid Distribution", fontsize=20)
    plt.legend(title=None, fontsize=16)
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
