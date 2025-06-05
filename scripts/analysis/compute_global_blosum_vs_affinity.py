#!/usr/bin/env python3
"""
End-to-end script to compute chain-aware “global‐reference” BLOSUM 90 similarity scores for mutated antibody/nanobody sequences
(using sequences from CSVs), correlate those scores with experimental affinities (−log(KD)), and plot the results.

Key differences from the PSSM version:
  - We supply three raw “global” sequences (heavy, kappa light, lambda light), each AHo‐numbered via ANARCI.
  - We build a global_map {pos_key: aa} for each chain.  For each mutant, we use the same union‐mask logic as before,
    but now sum BLOSUM90 scores at masked positions between the mutant’s aligned AA and the corresponding global AA.
  - The light chain is handled chain‐aware (“K” vs. “L”).
  - All file paths and dataset lists mirror the original PSSM script.

Usage:
  1) Fill in GLOBAL_HEAVY_RAW, GLOBAL_KAPPA_RAW, GLOBAL_LAMBDA_RAW with your ungapped one-letter global reference sequences.
  2) Run: python3 compute_global_blosum_vs_affinity.py
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
from Bio.Align import substitution_matrices
from matplotlib import rcParams

rcParams['font.weight'] = 'bold'

# ─────────────────────────────────────────────────────────────────────────────
#                              USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEBUG = True  # Set to False to disable debug prints

BLOSUM_NUM=45

# 1) Fill in your raw global one-letter (ungapped) sequences:
GLOBAL_DATA = 'OAS'
GLOBAL_HEAVY_RAW   = "QVQLVQS-GPGLVKPGESLSLSCKASGYSFSS------YAWSWVRQAPGKGLEWMGRIYP----SGDTNYAPSLKGRVTISRDTSKNTAYLQLSSLTAEDTAVYYCARDGGG---------------------YYFDYWGQGTLVTVSS"
GLOBAL_KAPPA_RAW   = "DIVMTQSPDSLSVSPGERATISCRASQSIS--------SYLAWYQQKPGQAPKLLIY---------ASTRASGVPSRFSGSGSG--TDFTLTISSLEAEDFAVYYCQQYSS-----------------------LPLTFGQGTKVEIK-"
GLOBAL_LAMBDA_RAW  = "QSVLTQP-PSVSVSPGQTVTLTCTGSSGSVGS------YYVSWYQQKPGQAPRLLIYE--------DNNRPSGVPDRFSGSKSG--NTASLTISGLQAEDEADYYCQSYDSS----------------------SAWVFGGGTKLTVL-"

# GLOBAL_DATA = 'SABDAB'
# GLOBAL_HEAVY_RAW   = "QVQLVES-GGGLVQPGGSLRLSC-AASG-FTFSS-----------YAMHWVRQ-AP-------G-----------KGLEWVGYISP----------GGSTYYADSVKGRFTISRDNS--------KNTAYLQMNSLRSEDTAVYYCARGGGY---------------------YYFDYWGQGT-LVTVSS"
# GLOBAL_KAPPA_RAW   = "DIVMTQSPSSLSASVGDRVTITCRAS--QSIS--------SYLAWYQQKPGQAPKLLIYG--------ASNLASGVPSRFSGSGSG----TDFTLTISSLQPEDFATYYCQQYYS-----------------------YPYTFGQGTKLEIKR"
# GLOBAL_LAMBDA_RAW  = "-SVLTQP-PSVSGSPGQTVTISCTGS--SNIGS-----NYVSWYQQ-K-PGQAPKLLIYD--------NSNRPSGVPDRFSGSKSG---TTASLTISGLQAEDEADYYCQSWDS-----------------------SPWVFGGGTKLTVLG"



# Paths (same as original scripts):
ROOT_DIR            = "/home/jovyan/abflow-datavol/"
RESULTS_DIR         = os.path.join(ROOT_DIR, "results")
EXPERIMENTAL_DATA   = "/home/jovyan/mlab-de-novo-data-4t/data/experimental_data/"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#                            BLOSUM 90 MATRIX LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Load BLOSUM90 from Biopython
blosum45 = substitution_matrices.load(f"BLOSUM{BLOSUM_NUM}")

def get_blosum_score(aa1: str, aa2: str) -> float:
    """
    Return the BLOSUM90 score for aa1 vs. aa2 (both single-letter).
    If the pair is not in the matrix, return 0.
    """
    try:
        return blosum45[(aa1, aa2)]
    except KeyError:
        try:
            return blosum45[(aa2, aa1)]
        except KeyError:
            return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#                           GLOBAL REFERENCE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def build_global_map(raw_seq: str, expected_chain: str) -> dict:
    """
    Given a raw one-letter antibody sequence and the expected chain type ('H','K','L'),
    run ANARCI (scheme='aho') to get AHo numbering, and build a map { pos_key: aa }.
    pos_key is int (e.g. 25) if no insertion, or str (e.g. "25A") if insertion.
    """
    raw_seq = raw_seq.replace('-', '')

    numbering, chain_type = number(raw_seq, scheme='aho')
    # if chain_type != expected_chain:
    #     raise ValueError(f"ANARCI mis-identified global {expected_chain} as '{chain_type}'")
    global_map = {}
    for (num, icode), aa in numbering:
        # Build pos_key
        if str(icode).strip() == "":
            pos_key = num
        else:
            pos_key = f"{num}{icode}".strip()
        global_map[pos_key] = aa
    if DEBUG:
        sample = list(global_map.items())[:5]
        print(f"[DEBUG] Built global_map for {expected_chain}, sample: {sample} …")
    return global_map

# Build the three global maps:
GLOBAL_HEAVY_MAP  = build_global_map(GLOBAL_HEAVY_RAW, 'H')
GLOBAL_KAPPA_MAP  = build_global_map(GLOBAL_KAPPA_RAW, 'K')
GLOBAL_LAMBDA_MAP = build_global_map(GLOBAL_LAMBDA_RAW, 'L')

# ─────────────────────────────────────────────────────────────────────────────
#                           IMPORT & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

from abflow.constants import aa1_name_to_index

INV_AA_MAP = {v:k for k,v in aa1_name_to_index.items()}
VALID_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# ─────────────────────────────────────────────────────────────────────────────
#                    CORRELATION & PLOTTING UTILITIES
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
#                           SEQUENCE CLEANING UTILS
# ─────────────────────────────────────────────────────────────────────────────

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
#                    PARENT & MUTANT SEQUENCE PREPARATION
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
        'heavy_numbering': list of ((resnum, icode), aa_char) for parent heavy (length M_h),
        'light_numbering': list of ((resnum, icode), aa_char) for parent light (length M_l),
        'heavy_chain':     'H',
        'light_chain':     'K' or 'L' (None if no light),
        'seq_tensors':     list of aligned mutant strings (length = M_h+M_l),
        'mut_numberings':  list of tuples (mut_heavy_numbering, mut_light_numbering),
        'mask':            torch.BoolTensor (N × (M_h+M_l)) union mask in AHo‐space,
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

    # 1) Number parent heavy via ANARCI
    heavy_numbering, heavy_chain = number(heavy_parent, scheme='aho')
    if heavy_chain != "H":
        raise ValueError(f"ANARCI did not recognize parent heavy as 'H' (got '{heavy_chain}')")
    M_h = len(heavy_numbering)

    # 2) Number parent light if present
    if light_parent:
        light_numbering, light_chain = number(light_parent, scheme='aho')
        if light_chain not in ("K","L"):
            raise ValueError(f"ANARCI did not recognize parent light as 'K' or 'L' (got '{light_chain}')")
        M_l = len(light_numbering)
    else:
        light_numbering, light_chain = [], None
        M_l = 0

    # Load mutated CSV
    target_df = pd.read_csv(target_csv)
    parent_concat = heavy_parent + light_parent

    seq_tensors = []
    KD_values   = []
    binder_list = []
    mut_numberings = []

    # Build union masks in AHo-space
    union_heavy_mask = np.zeros(M_h, dtype=bool)
    union_light_mask = np.zeros(M_l, dtype=bool)

    for idx, row in target_df.iterrows():
        # Determine KD value depending on dataset
        if target_name in ['absci_her2_sc','c5','il17a','tslp','acvr2b','fxi','il36r','tnfrsf9']:
            kd_val = row.get('KD (nM)', np.nan)
        elif target_name in ['nature_il7','lox1','scf']:
            kd_val = row.get('IC50 (M)', np.nan)
        elif target_name == 'absci_her2_zs':
            kd_val = row.get('-log(KD (M))', np.nan)
        elif target_name == 'tweak':
            kd_val = row.get('DDG', np.nan)
        else:
            kd_val = row.get('KD', np.nan)

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

        raw_mut_heavy = mutant_concat[:len(heavy_parent)]
        raw_mut_light = mutant_concat[len(heavy_parent):] if light_parent else ""

        # 3) Renumber the mutant heavy & light via ANARCI
        mut_heavy_numbering, mut_heavy_chain = number(raw_mut_heavy, scheme='aho')
        if mut_heavy_chain != heavy_chain or len(mut_heavy_numbering) != M_h:
            if DEBUG:
                print(f"[DEBUG] Skipping mutant {idx}: heavy numbering mismatch")
            continue

        if light_parent:
            mut_light_numbering, mut_light_chain = number(raw_mut_light, scheme='aho')
            if mut_light_chain != light_chain or len(mut_light_numbering) != M_l:
                if DEBUG:
                    print(f"[DEBUG] Skipping mutant {idx}: light numbering mismatch")
                continue
        else:
            mut_light_numbering, mut_light_chain = [], None

        # 4) Build this mutant’s mask in AHo-space
        this_mask_h = np.zeros(M_h, dtype=bool)
        for i in range(M_h):
            parent_aa = heavy_numbering[i][1]
            mut_aa    = mut_heavy_numbering[i][1]
            if parent_aa != mut_aa:
                this_mask_h[i] = True

        if light_parent:
            this_mask_l = np.zeros(M_l, dtype=bool)
            for i in range(M_l):
                parent_aa = light_numbering[i][1]
                mut_aa    = mut_light_numbering[i][1]
                if parent_aa != mut_aa:
                    this_mask_l[i] = True
        else:
            this_mask_l = np.zeros(0, dtype=bool)

        # 5) Update the union masks
        union_heavy_mask |= this_mask_h
        if light_parent:
            union_light_mask |= this_mask_l

        # 6) Store the aligned mutant string (heavy + light) from renumbered lists
        aligned_heavy = "".join([aa for ((num, icode), aa) in mut_heavy_numbering])
        if light_parent:
            aligned_light = "".join([aa for ((num, icode), aa) in mut_light_numbering])
        else:
            aligned_light = ""
        aligned_full = aligned_heavy + aligned_light  # length = M_h + M_l
        seq_tensors.append(aligned_full)

        # Store numbering lists for scoring later
        mut_numberings.append((mut_heavy_numbering, mut_light_numbering))

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

        KD_values.append(log_kd)
        if 'Binder' in row and pd.notnull(row['Binder']):
            binder_list.append(int(row['Binder']))

    # Build the final union mask tensor (N × (M_h+M_l))
    if not seq_tensors:
        mask_tensor = torch.zeros((0, M_h + M_l), dtype=torch.bool)
    else:
        union_full = np.concatenate([union_heavy_mask, union_light_mask])
        mask_tensor = torch.tensor(union_full, dtype=torch.bool).unsqueeze(0).repeat(len(seq_tensors), 1)

    return {
        'heavy_parent':    heavy_parent,
        'light_parent':    light_parent,
        'heavy_numbering': heavy_numbering,
        'light_numbering': light_numbering,
        'heavy_chain':     heavy_chain,
        'light_chain':     light_chain,
        'seq_tensors':     seq_tensors,        # list of aligned strings (length M_h+M_l)
        'mut_numberings':  mut_numberings,     # list of (mut_heavy_numbering, mut_light_numbering)
        'mask':            mask_tensor,        # torch.BoolTensor (N × (M_h+M_l))
        'KD_values':       KD_values,
        'Binder':          binder_list
    }

# ─────────────────────────────────────────────────────────────────────────────
#              COMPUTE GLOBAL BLOSUM 90 SCORE FOR A SINGLE CHAIN
# ─────────────────────────────────────────────────────────────────────────────

def compute_blosum_score_for_chain(mut_numbering: list, mask_chain: np.ndarray, global_map: dict) -> float:
    """
    Given:
      - mut_numbering: list of ((resnum, icode), aa_char) from ANARCI for this mutant chain
      - mask_chain: boolean numpy array of length = len(mut_numbering); True where we score
      - global_map: { pos_key: aa_char } for the global reference of this chain
    Returns:
      - sum of BLOSUM90 scores over masked positions for this chain.
    """
    total = 0.0
    for idx, do_score in enumerate(mask_chain):
        if not do_score:
            continue
        (num, icode), mut_aa = mut_numbering[idx]
        if mut_aa == '-':
            continue
        # Build pos_key
        if str(icode).strip() == "":
            pos_key = num
        else:
            pos_key = f"{num}{icode}".strip()
        if pos_key not in global_map:
            continue
        glob_aa = global_map[pos_key]
        if glob_aa == '-':
            continue
        total += get_blosum_score(glob_aa, mut_aa)
    return total

# ─────────────────────────────────────────────────────────────────────────────
#       MAIN EVALUATION FUNCTION: GLOBAL BLOSUM 90‐BASED CORRELATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_with_global_blosum(target: str, results_dir: str, compute_auc: bool=False):
    """
    For a given dataset name (e.g. 'absci_her2_sc'), loads parent CSV and mutated CSV,
    computes global BLOSUM90 scores for each mutant (heavy + light),
    computes correlations vs. −log(KD), and writes plots + summary CSV.
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
    seq_aligned     = data['seq_tensors']    # list of aligned strings (length M_h+M_l)
    mut_numberings  = data['mut_numberings'] # list of (mut_heavy_numbering, mut_light_numbering)
    mask_tensor     = data['mask']           # (N × (M_h+M_l))
    KD_vals         = np.array(data['KD_values'])
    binder_labels   = np.array(data['Binder']) if data['Binder'] else None

    M_h = len(heavy_numbering)
    M_l = len(light_numbering)
    N = len(seq_aligned)
    blosum_scores = np.zeros(N, dtype=float)

    for i in range(N):
        # Extract this mutant’s numbering lists and mask
        mut_heavy_numbering, mut_light_numbering = mut_numberings[i]
        mask_i = mask_tensor[i].numpy()             # union mask in AHo-space
        mask_heavy = mask_i[:M_h]
        mask_light = mask_i[M_h:] if M_l > 0 else np.zeros(0, dtype=bool)

        # Compute heavy chain BLOSUM score
        score_h = compute_blosum_score_for_chain(mut_heavy_numbering, mask_heavy, GLOBAL_HEAVY_MAP)

        # Compute light chain BLOSUM score, chain-aware
        if light_chain == 'K':
            score_l = compute_blosum_score_for_chain(mut_light_numbering, mask_light, GLOBAL_KAPPA_MAP)
        elif light_chain == 'L':
            score_l = compute_blosum_score_for_chain(mut_light_numbering, mask_light, GLOBAL_LAMBDA_MAP)
        else:
            score_l = 0.0

        blosum_scores[i] = score_h + score_l
        if DEBUG:
            print(f"[DEBUG] Mutant {i}: BLOSUM_heavy={score_h:.4g}, BLOSUM_light={score_l:.4g}, total={blosum_scores[i]:.4g}")

    # Correlations + plotting
    if target == 'napi2b':
        target_name = 'AZ-tg1'
    elif target == 'sonic':
        target_name = 'AZ-tg2'
    else:
        target_name = target

    plot_path = os.path.join(results_dir, f"{GLOBAL_DATA}_globalBLOSUM{BLOSUM_NUM}_vs_affinity_{target_name}.pdf")
    plot_prop_correlation(blosum_scores, KD_vals, f"BLOSUM{BLOSUM_NUM}", plot_path)
    print(f"[{target}] Saved BLOSUM{BLOSUM_NUM} vs KD plot → {plot_path}")

    # Summary stats
    τ, pτ = kendalltau(blosum_scores, KD_vals)
    ρ, pρ = spearmanr(blosum_scores, KD_vals)
    sτ, sρ = get_significance(pτ), get_significance(pρ)

    summary_record = {
        'dataset':      target_name,
        'metric':       f"BLOSUM{BLOSUM_NUM}",
        'spearman_rho': ρ,
        'p_spearman':   pρ,
        'spearman_sig': sρ,
        'kendall_tau':  τ,
        'p_kendall':    pτ,
        'kendall_sig':  sτ
    }

    # ROC‐AUC if requested
    if compute_auc and binder_labels is not None and len(binder_labels) == N:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            auc_score = roc_auc_score(binder_labels, blosum_scores)
            fpr, tpr, _ = roc_curve(binder_labels, blosum_scores)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
            plt.plot([0,1],[0,1],'k--', label='Chance')
            plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
            plt.title(f'ROC (BLOSUM90) for {target_name}', fontsize=16, fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(True)
            roc_path = plot_path.replace('.pdf', '_roc_auc.pdf')
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            print(f"[{target}] Saved ROC AUC plot → {roc_path}")
        except Exception as e:
            print(f"[{target}] Error computing ROC AUC: {e}")

    # Save summary CSV for this dataset
    df_sum = pd.DataFrame([summary_record])
    csv_path = os.path.join(results_dir, f"{GLOBAL_DATA}_globalBLOSUM{BLOSUM_NUM}_correlation_{target_name}.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"[{target}] Saved summary CSV → {csv_path}")

    return summary_record

# ─────────────────────────────────────────────────────────────────────────────
#                                   MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Define your datasets exactly as in the original scripts
    dataset1 = ['absci_her2_zs','absci_her2_sc','nature_hel','nature_il7','nature_her2','napi2b','sonic']
    dataset2 = ['c5','il17a','tslp','fxi','il36r','tnfrsf9','acvr2b']
    complex_list = dataset1 + dataset2

    all_summary = []

    for target in complex_list:
        summary = evaluate_dataset_with_global_blosum(target, RESULTS_DIR, compute_auc=False)
        all_summary.append(summary)

    # Save combined summary
    combined_df = pd.DataFrame(all_summary)
    combined_df['spearman'] = combined_df.apply(lambda r: f"{r['spearman_rho']:.3f}{r['spearman_sig']}", axis=1)
    combined_df['kendall']  = combined_df.apply(lambda r: f"{r['kendall_tau']:.3f}{r['kendall_sig']}", axis=1)

    out = combined_df[['dataset','metric','spearman','p_spearman','kendall','p_kendall']]
    all_csv = os.path.join(RESULTS_DIR, f"{GLOBAL_DATA}_all_datasets_globalBLOSUM{BLOSUM_NUM}_summary.csv")
    out.to_csv(all_csv, index=False)
    print(f"Saved combined summary → {all_csv}")
