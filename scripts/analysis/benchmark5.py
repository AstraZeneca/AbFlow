#!/usr/bin/env python3
"""
End-to-end script to compute chain-aware PSSM scores for mutated antibody/nanobody sequences
(using sequences from CSVs), correlate those scores with experimental affinities (−log(KD)),
and plot the results.

Key corrections:
  - Use ANARCI’s correct signature: number(sequence, scheme='aho') → (numbering_list, chain_type).
  - Build a single “union” mask across all mutants (True wherever any mutant differs from parent).
  - Pre-clean parent and mutant strings to remove any non-ACDEFGHIKLMNPQRSTVWY characters.
  - Computes PSSM scores chain-aware (heavy vs. κ vs. λ).
  - Supports optional ROC-AUC if 'Binder' column exists.
  - DEBUG prints show mutation positions and PSSM lookups.

DEBUG prints are on by default. Set DEBUG=False to silence them.
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, kendalltau
from anarci import number
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import rcParams
import matplotlib.ticker as mticker
rcParams['font.weight'] = 'bold'
# ─────────────────────────────────────────────────────────────────────────────
#                              USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEBUG = True  # Set to False to disable debug prints

# Correct PSSM file paths
HEAVY_PSSM_PATH  = "/home/jovyan/abflow-datavol/github_repos/AApull_request/AbFlow/scripts/analysis/OAS-Human_HeavyPWM_frequencypssm.txt"
KAPPA_PSSM_PATH  = "/home/jovyan/abflow-datavol/github_repos/AApull_request/AbFlow/scripts/analysis/OAS-Human_KappaPWM_frequencypssm.txt"
LAMBDA_PSSM_PATH = "/home/jovyan/abflow-datavol/github_repos/AApull_request/AbFlow/scripts/analysis/OAS-Human_LambdaPWM_frequencypssm.txt"

# Paths related to AbFlow (for potential future use)
ROOT_DIR            = "/home/jovyan/abflow-datavol/"
EXPERIMENT_NAME     = "abflow_may17_2dmask_CDR3ONLY_Fixed_numAtomAndMasking_diffab_sabdab_sequence_backbone"
EPOCH_NUM           = "epoch=69"
DEVICE              = "cuda:0"
RESULTS_DIR         = os.path.join(ROOT_DIR, "results")
EXPERIMENT_DIR      = os.path.join(ROOT_DIR, "checkpoints", EXPERIMENT_NAME)
CONFIG_PATH         = os.path.join(EXPERIMENT_DIR, "config.yaml")

# Experimental data base directory
EXPERIMENTAL_DATA   = "/home/jovyan/mlab-de-novo-data-4t/data/experimental_data/"

# Other parameters
COMPUTE_AUC = False  # Set True to compute ROC‐AUC if Binder labels exist

os.makedirs(RESULTS_DIR, exist_ok=True)

if DEBUG:
    print(f"[DEBUG] DEVICE             = {DEVICE}")
    print(f"[DEBUG] RESULTS_DIR        = {RESULTS_DIR}")
    print(f"[DEBUG] HEAVY_PSSM_PATH    = {HEAVY_PSSM_PATH}")
    print(f"[DEBUG] KAPPA_PSSM_PATH    = {KAPPA_PSSM_PATH}")
    print(f"[DEBUG] LAMBDA_PSSM_PATH   = {LAMBDA_PSSM_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
#                                   IMPORT UTILS
# ─────────────────────────────────────────────────────────────────────────────

import sys
sys.path.append(os.path.join(ROOT_DIR, "github_repos", "AApull_request", "AbFlow"))

from abflow.constants import aa1_name_to_index

# ─────────────────────────────────────────────────────────────────────────────
#                             PSSM LOADING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_pssm(txt_path: str) -> dict[str, dict[str, float]]:
    """
    Load a PSSM file (first line “Nseqs=…| …”) into a nested dict:
      { aho_pos_str: { 'A': float, 'C': float, …, 'Y': float }, … }.
    """
    with open(txt_path, "r") as f:
        first_line = f.readline().strip()
    all_cols = first_line.split()
    if "AHo" in all_cols:
        idx = all_cols.index("AHo")
        all_cols[idx] = "AHo numbering"
        del all_cols[idx+1]
    if DEBUG:
        print(f"[DEBUG] Parsed columns for {txt_path}: {all_cols[:6]} … {all_cols[-5:]}")

    df = pd.read_csv(
        txt_path,
        delim_whitespace=True,
        header=None,
        skiprows=1,
        engine="python"
    )
    df.columns = all_cols
    df = df.set_index("AHo numbering")
    aa_columns = [c for c in df.columns if len(c) == 1 and c in list("ACDEFGHIKLMNPQRSTVWY")]
    if DEBUG:
        print(f"[DEBUG] Recognized AA columns in {txt_path}: {aa_columns}")
        print(f"[DEBUG] {txt_path} dimensions: {df.shape[0]} rows × {len(aa_columns)} AA cols")
    df_aa = df[aa_columns].astype(float)
    nested = df_aa.to_dict(orient="index")
    if DEBUG:
        sample = list(nested.items())[:3]
        print(f"[DEBUG] Sample entries from {txt_path}:")
        for pos, row in sample:
            print(f"   {pos}: {{ {', '.join(f'{aa}:{row[aa]:.4g}' for aa in aa_columns[:3])} … }}")
    return nested

heavy_pssm  = load_pssm(HEAVY_PSSM_PATH)
kappa_pssm  = load_pssm(KAPPA_PSSM_PATH)
lambda_pssm = load_pssm(LAMBDA_PSSM_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#                        SEQUENCE ↔ ONE‐LETTER DECODING UTILS
# ─────────────────────────────────────────────────────────────────────────────

INV_AA_MAP = {v: k for k, v in aa1_name_to_index.items()}
VALID_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(seq: str) -> str:
    """
    Uppercase and remove any character not in the 20 standard amino acids.
    """
    s = str(seq).upper()
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', s)
    if DEBUG and cleaned != s:
        removed = set(s) - set(cleaned)
        print(f"[DEBUG] clean_sequence: removed {removed} from '{seq}' → '{cleaned}'")
    return cleaned

# ─────────────────────────────────────────────────────────────────────────────
#                           PSSM‐BASED SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pssm_score_for_seq(numbering: list[tuple[tuple[int,str], str]],
                               seq: str, mask: np.ndarray,
                               chain_type: str,
                               debug_prefix: str="") -> float:
    """
    Given:
      - numbering: list of ((resnum, icode), aa_char) tuples from ANARCI for this chain
      - seq: one-letter string for the variable region (H or L) exactly as passed to ANARCI
      - mask: boolean numpy array of length len(seq) (True at positions to score)
      - chain_type: 'H', 'K', or 'L'
      - debug_prefix: string to prefix debug prints
    Returns:
      - sum of PSSM frequencies over masked positions for this chain.
    """
    total = 0.0

    for idx, do_score in enumerate(mask):
        if not do_score:
            continue
        (num, icode), aa_char = numbering[idx]
        pos_str = f"{num}{icode}" if icode != " " else num
        mut_aa = seq[idx]
        val = 0.0
        if chain_type == "H":
            if pos_str in heavy_pssm and mut_aa in heavy_pssm[pos_str]:
                val = heavy_pssm[pos_str][mut_aa]

        elif chain_type == "K":
            if pos_str in kappa_pssm and mut_aa in kappa_pssm[pos_str]:
                val = kappa_pssm[pos_str][mut_aa]
        elif chain_type == "L":
            if pos_str in lambda_pssm and mut_aa in lambda_pssm[pos_str]:
                val = lambda_pssm[pos_str][mut_aa]
        if DEBUG and debug_prefix:
            print(f"{debug_prefix} idx={idx}, pos={pos_str}, aa={mut_aa}, val={val}")
        total += val
    return total

# ─────────────────────────────────────────────────────────────────────────────
#                          CORRELATION & PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_significance(p_value: float) -> str:
    if p_value < 1e-4:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''

def plot_prop_correlation(x: np.ndarray, y: np.ndarray, xlabel: str, save_path: str):
    """
    Joint scatter + KDE plot for x vs. y, with Kendall τ and Spearman ρ.
    """
    τ, pτ = kendalltau(x, y)
    ρ, pρ = spearmanr(x, y)
    sτ, sρ = get_significance(pτ), get_significance(pρ)

    df = pd.DataFrame({xlabel: x, '-log(KD)': y})
    g = sns.jointplot(x=xlabel, y='-log(KD)', data=df, kind='scatter', color='purple')
    g.plot_joint(sns.kdeplot, fill=True, levels=6, alpha=0.4, color='purple')
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()
    g.fig.set_size_inches(6, 6)
    plt.grid(color='gray', linestyle='dashed', zorder=0)

    red_patch  = mpatches.Patch(color='red',  label=fr'Kendall τ: {τ:.2f}{sτ}')
    blue_patch = mpatches.Patch(color='blue', label=fr'Spearman ρ: {ρ:.2f}{sρ}')
    leg = plt.legend(handles=[red_patch, blue_patch], loc='upper left',
                     handlelength=0, handletextpad=0, fancybox=True, fontsize=12)
    for h in leg.legend_handles:
        h.set_visible(False)

    ax = g.ax_joint
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(r'$-\log(K_D)$', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    class ScalarFormatterForceFormat(plt.matplotlib.ticker.ScalarFormatter):
        def _set_format(self, vmin=None, vmax=None):
            self.format = '%.1f'
    fmt = ScalarFormatterForceFormat(useMathText=False)
    fmt.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(fmt)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(12)
    ax.xaxis.get_offset_text().set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#                    SEQUENCE + AFFINITY GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def convert_kd_to_log_kd(kd_M: float) -> float:
    if kd_M <= 0:
        return np.nan
    return -np.log10(kd_M)

def generate_sequences_and_kd(parental_csv: str, target_csv: str,
                              target_name: str = None, compute_auc: bool = False):
    """
    Returns:
      {
        'heavy_parent':    cleaned one-letter heavy string,
        'light_parent':    cleaned one-letter light string ("" if none),
        'heavy_numbering': list of ((resnum, icode), aa_char) for parent heavy,
        'light_numbering': list of ((resnum, icode), aa_char) for parent light (empty if none),
        'heavy_chain':     'H',
        'light_chain':     'K' or 'L' (None if no light),
        'seq_tensors':     list of concatenated heavy+light torch.Tensors for mutants,
        'mask':            torch.BoolTensor (1 × L) union mask across all mutants,
        'KD_values':       list of -log10(KD) floats,
        'Binder':          list of binder labels if present (else empty).
      }
    """
    parental_df = pd.read_csv(parental_csv)
    raw_heavy = parental_df["Heavy"].iloc[0]
    heavy_parent = clean_sequence(raw_heavy)

    # Handle missing or NaN Light
    if "Light" in parental_df.columns and pd.notnull(parental_df["Light"].iloc[0]):
        raw_light = parental_df["Light"].iloc[0]
        light_parent = clean_sequence(raw_light)
    else:
        raw_light = ""
        light_parent = ""
    if DEBUG:
        print(f"[DEBUG] raw_heavy: '{raw_heavy}' → '{heavy_parent}'")
        print(f"[DEBUG] raw_light: '{raw_light}' → '{light_parent}'")

    # Number parent heavy via ANARCI
    heavy_numbering, heavy_chain = number(heavy_parent, scheme='aho')
    if heavy_chain != "H":
        raise ValueError(f"ANARCI did not recognize parent heavy as 'H' (got '{heavy_chain}')")

    # Number parent light if present
    if light_parent:
        light_numbering, light_chain = number(light_parent, scheme='aho')
        if light_chain not in ("K","L"):
            raise ValueError(f"ANARCI did not recognize parent light as 'K' or 'L' (got '{light_chain}')")
    else:
        light_numbering, light_chain = [], None

    # Load mutated CSV
    target_df = pd.read_csv(target_csv)
    parent_concat = heavy_parent + light_parent
    parent_tensor = torch.tensor([aa1_name_to_index[aa] for aa in parent_concat], dtype=torch.long)

    seq_tensors = []
    KD_values = []
    binder_list = []

    # Build union mask (as Python list of booleans) initially all False
    union_mask = np.zeros(len(parent_concat), dtype=bool)

    for idx, row in target_df.iterrows():
        # Determine KD value depending on dataset
        if target_name in ['absci_her2_sc','c5','il17a','tslp','acvr2b','fxi','il36r','tnfrsf9']:
            kd_val = row['KD (nM)']
        elif target_name in ['nature_il7','lox1','scf']:
            kd_val = row['IC50 (M)']
        elif target_name == 'absci_her2_zs':
            kd_val = row['-log(KD (M))']
        elif target_name == 'tweak':
            kd_val = row['DDG']
        else:
            kd_val = row['KD']

        if not compute_auc and pd.isnull(kd_val):
            continue

        # Build mutant from parent
        seq_list = list(parent_concat)
        use_only_cdrs = any(c in parental_df.columns for c in ['HCDR1','HCDR2','HCDR3','LCDR1','LCDR2','LCDR3'])

        if use_only_cdrs:
            for cdr in ['HCDR1','HCDR2','HCDR3','LCDR1','LCDR2','LCDR3']:
                if cdr in row and pd.notnull(row[cdr]):
                    raw_targ = row[cdr]
                    targ_seq = clean_sequence(raw_targ)
                    parent_seq = clean_sequence(str(parental_df[cdr].iloc[0]))
                    if len(parent_seq) != len(targ_seq):
                        seq_list = None
                        break
                    start_idx = parent_concat.find(parent_seq)
                    for offset, aa in enumerate(targ_seq):
                        seq_list[start_idx+offset] = aa
            if seq_list is None:
                continue
        else:
            # Heavy replacement
            if "Heavy" in row and pd.notnull(row["Heavy"]):
                raw_targ = row["Heavy"]
                targ_seq = clean_sequence(raw_targ)
                if len(heavy_parent) != len(targ_seq):
                    continue
                for offset, aa in enumerate(targ_seq):
                    seq_list[offset] = aa
            # Light replacement, only if light_parent exists
            if light_parent and "Light" in row and pd.notnull(row["Light"]):
                raw_targ = row["Light"]
                targ_seq = clean_sequence(raw_targ)
                if len(light_parent) != len(targ_seq):
                    continue
                start_idx = len(heavy_parent)
                for offset, aa in enumerate(targ_seq):
                    seq_list[start_idx+offset] = aa

        mutant_concat = "".join(seq_list)
        invalid = set(mutant_concat) - VALID_AA_SET
        if invalid:
            if DEBUG:
                print(f"[DEBUG] Skipping mutant with invalid letters: {invalid}")
            continue

        mutant_tensor = torch.tensor([aa1_name_to_index[aa] for aa in mutant_concat], dtype=torch.long)
        diff = (mutant_tensor != parent_tensor).numpy()
        union_mask |= diff  # update union mask

        # Compute −log(KD)
        try:
            if target_name in ['absci_her2_sc','c5','il17a','tslp','acvr2b','fxi','il36r','tnfrsf9']:
                log_kd = convert_kd_to_log_kd(1e-9 * float(kd_val))
            elif target_name in ['nature_il7','lox1','scf']:
                log_kd = convert_kd_to_log_kd(float(kd_val))
            elif target_name == 'absci_her2_zs':
                log_kd = float(kd_val)
            elif target_name == 'tweak':
                log_kd = float(kd_val)
            else:
                log_kd = convert_kd_to_log_kd(float(kd_val))
        except:
            continue

        seq_tensors.append(mutant_tensor)
        KD_values.append(log_kd)
        if 'Binder' in row and pd.notnull(row['Binder']):
            binder_list.append(int(row['Binder']))

    if not seq_tensors:
        # No valid mutants
        mask_tensor = torch.zeros((0, len(parent_concat)), dtype=torch.bool)
    else:
        # Build a single mask tensor (1 × L) and repeat for all mutants
        mask_tensor = torch.tensor(union_mask, dtype=torch.bool).unsqueeze(0).repeat(len(seq_tensors), 1)

    return {
        'heavy_parent':    heavy_parent,
        'light_parent':    light_parent,
        'heavy_numbering': heavy_numbering,
        'light_numbering': light_numbering,
        'heavy_chain':     heavy_chain,
        'light_chain':     light_chain,
        'seq_tensors':     seq_tensors,
        'mask':            mask_tensor,   # (N × L)
        'KD_values':       KD_values,
        'Binder':          binder_list
    }

# ─────────────────────────────────────────────────────────────────────────────
#          MAIN EVALUATION FUNCTION: PSSM‐BASED CORRELATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_with_pssm(target: str, results_dir: str, compute_auc: bool=False):
    """
    For a given dataset name (e.g. 'absci_her2_sc'), loads parent CSV and mutated CSV,
    computes PSSM scores for each mutant, computes correlations vs. −log(KD),
    and writes plots + summary CSV.
    """
    parent_csv  = os.path.join(EXPERIMENTAL_DATA, target, f"{target}_parent.csv")
    mutated_csv = os.path.join(EXPERIMENTAL_DATA, target, f"{target}.csv")

    if DEBUG:
        print(f"[DEBUG] Processing dataset: {target}")
        print(f"[DEBUG] Parent CSV  = {parent_csv}")
        print(f"[DEBUG] Mutated CSV = {mutated_csv}")

    data = generate_sequences_and_kd(parent_csv, mutated_csv, target_name=target, compute_auc=compute_auc)
    heavy_parent    = data['heavy_parent']
    light_parent    = data['light_parent']
    heavy_numbering = data['heavy_numbering']
    light_numbering = data['light_numbering']
    heavy_chain     = data['heavy_chain']    # 'H'
    light_chain     = data['light_chain']    # 'K', 'L', or None
    seq_tensors     = data['seq_tensors']    # list of Tensors
    mask_tensor     = data['mask']           # (N × L)
    KD_vals         = np.array(data['KD_values'])
    binder_labels   = np.array(data['Binder']) if data['Binder'] else None

    h_len = len(heavy_parent)
    l_len = len(light_parent)
    N = len(seq_tensors)
    pssm_scores = np.zeros(N, dtype=float)

    for i in range(N):
        mask_i = mask_tensor[i].numpy()  # union mask
        mutant_concat = seq_tensors[i]
        mutant_str = "".join(INV_AA_MAP[int(idx)] for idx in mutant_concat.tolist())
        heavy_mut = mutant_str[:h_len]
        light_mut = mutant_str[h_len:] if l_len > 0 else ""

        mask_heavy = mask_i[:h_len]
        mask_light = mask_i[h_len:] if l_len > 0 else np.array([], dtype=bool)

        if DEBUG:
            print(f"[DEBUG] Evaluating mutant {i}: heavy_mut='{heavy_mut}', light_mut='{light_mut}'")
            print(f"[DEBUG] Union mask heavy: {mask_heavy}, light: {mask_light}")

        sc_h = compute_pssm_score_for_seq(heavy_numbering, heavy_mut, mask_heavy,
                                          chain_type=heavy_chain,
                                          debug_prefix=f"[DEBUG][Mut{i}][H] ")
        if light_chain:
            sc_l = compute_pssm_score_for_seq(light_numbering, light_mut, mask_light,
                                              chain_type=light_chain,
                                              debug_prefix=f"[DEBUG][Mut{i}][L] ")
        else:
            sc_l = 0.0

        pssm_scores[i] = sc_h + sc_l
        if DEBUG:
            print(f"[DEBUG] Mutant {i}: PSSM_heavy={sc_h:.4g}, PSSM_light={sc_l:.4g}, total={pssm_scores[i]:.4g}")

    summary_records = []

    # PSSM vs. KD
    τ, pτ = kendalltau(pssm_scores, KD_vals)
    ρ, pρ = spearmanr(pssm_scores, KD_vals)
    sτ, sρ = get_significance(pτ), get_significance(pρ)
    plot_path = os.path.join(results_dir, f"yPSSM_vs_affinity_{target}_{EPOCH_NUM}.pdf")
    plot_prop_correlation(pssm_scores, KD_vals, "PSSM", plot_path)
    print(f"[{target}] Saved PSSM vs KD plot → {plot_path}")
    summary_records.append({
        'dataset':      target,
        'metric':       "PSSM",
        'spearman_rho': ρ,
        'p_spearman':   pρ,
        'spearman_sig': sρ,
        'kendall_tau':  τ,
        'p_kendall':    pτ,
        'kendall_sig':  sτ
    })

    # ROC‐AUC if binder labels provided
    if compute_auc and binder_labels is not None and len(binder_labels) == N:
        try:
            auc_score = roc_auc_score(binder_labels, pssm_scores)
            fpr, tpr, _ = roc_curve(binder_labels, pssm_scores)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            plt.plot([0,1],[0,1],'k--', label='Chance')
            plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
            plt.title(f'ROC (PSSM) for {target}', fontsize=16, fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(True)
            roc_path = plot_path.replace('.pdf', '_roc_auc.pdf')
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            print(f"[{target}] Saved ROC AUC plot → {roc_path}")
        except Exception as e:
            print(f"[{target}] Error computing ROC AUC: {e}")

    # Save summary CSV for this dataset
    df_sum = pd.DataFrame(summary_records)
    csv_path = os.path.join(results_dir, f"y{target}_pssm_correlation_{EPOCH_NUM}.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"[{target}] Saved summary CSV → {csv_path}")

    return summary_records

# ─────────────────────────────────────────────────────────────────────────────
#                                   MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Define your datasets
    dataset1 = ['absci_her2_zs','absci_her2_sc','nature_hel','nature_il7','nature_her2','AZtg1','AZtg2']
    dataset2 = ['c5','il17a','tslp','fxi','il36r','tnfrsf9','acvr2b']
    complex_list = dataset1 + dataset2

    all_summary = []

    for target in complex_list:
        summary = evaluate_dataset_with_pssm(target, RESULTS_DIR, compute_auc=COMPUTE_AUC)
        all_summary.extend(summary)

    # Save combined summary
    combined_df = pd.DataFrame(all_summary)
    combined_df['spearman'] = combined_df.apply(lambda r: f"{r.spearman_rho:.3f}{r.spearman_sig}", axis=1)
    combined_df['kendall'] = combined_df.apply(lambda r: f"{r.kendall_tau:.3f}{r.kendall_sig}", axis=1)

    out = combined_df[['dataset','metric','spearman','p_spearman','kendall','p_kendall']]
    all_csv = os.path.join(RESULTS_DIR, f"yall_datasets_pssm_summary_{EPOCH_NUM}.csv")
    out.to_csv(all_csv, index=False)
    print(f"Saved combined summary → {all_csv}")

    # Pivot tables
    spearman_table = out.pivot(index='metric', columns='dataset', values='spearman')
    kendall_table  = out.pivot(index='metric', columns='dataset', values='kendall')
    spearman_csv   = os.path.join(RESULTS_DIR, f"yall_datasets_spearman_summary_{EPOCH_NUM}.csv")
    kendall_csv    = os.path.join(RESULTS_DIR, f"yall_datasets_kendall_summary_{EPOCH_NUM}.csv")
    spearman_table.to_csv(spearman_csv)
    kendall_table.to_csv(kendall_csv)
    print(f"Saved Spearman pivot → {spearman_csv}")
    print(f"Saved Kendall pivot  → {kendall_csv}")
