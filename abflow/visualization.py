import torch
import matplotlib.pyplot as plt
import numpy as np

from .constants import AminoAcid1

DARK_BLUE = "#1f77b4"
LIGHT_BLUE = "#aec7e8"
DARK_RED = "#d62728"
LIGHT_RED = "#ff9896"

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_scatter(
    pred: torch.Tensor,
    true: torch.Tensor,
    x_label="Pred",
    y_label="True",
    save_path: str = None,
):
    """
    Plot true data against pred data on the resolved region.

    Args:
        true (torch.Tensor): True data of shape (N_batch,).
        plddt (torch.Tensor): Pred data of shape (N_batch,).
        save_path (str, optional): Path to save the figure. Defaults to None.
    """

    true = np.array(true)
    pred = np.array(pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(pred, true, s=50, alpha=0.6, color=DARK_BLUE, edgecolor="black")

    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_plddt(plddt: torch.Tensor, lddt: torch.Tensor, save_path: str = None):
    """
    Plot LDDT against Average pLDDT on the resolved region.

    Args:
        lddt (torch.Tensor): LDDT scores of shape (N_batch,).
        plddt (torch.Tensor): pLDDT scores of shape (N_batch,).
        save_path (str, optional): Path to save the figure. Defaults to None.
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


def plot_aa_distribution(
    pred_seq: torch.Tensor, true_seq: torch.Tensor, save_path: str = None
):
    """
    Plot histogram distribution for predicted and ground truth amino acid sequences.

    Args:
        pred_seq (torch.Tensor): Predicted amino acid sequence of shape (N,).
        true_seq (torch.Tensor): Ground truth amino acid sequence of shape (N,).
        save_path (str, optional): Path to save the figure. Defaults to None.
    """

    pred_seq = np.array(pred_seq)
    true_seq = np.array(true_seq)

    bins = np.arange(21) - 0.5  # Create 21 bins for 20 amino acids + 1

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot for predicted sequence
    axs[0].hist(
        pred_seq, bins=bins, alpha=0.7, color=DARK_BLUE, edgecolor="black", density=True
    )
    axs[0].set_xlabel("Amino Acids", fontsize=25)
    axs[0].set_ylabel("Density", fontsize=25)
    axs[0].set_title("Designed Sequence Distribution", fontsize=25)
    axs[0].set_xticks(np.arange(20))
    axs[0].set_xticklabels([aa for aa in AminoAcid1.__members__.keys()], fontsize=14)
    axs[0].grid(True)

    # Plot for ground truth sequence
    axs[1].hist(
        true_seq, bins=bins, alpha=0.7, color=DARK_RED, edgecolor="black", density=True
    )
    axs[1].set_xlabel("Amino Acids", fontsize=25)
    axs[1].set_title("True Sequence Distribution", fontsize=25)
    axs[1].set_xticks(np.arange(20))
    axs[1].set_xticklabels([aa for aa in AminoAcid1.__members__.keys()], fontsize=14)
    axs[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
