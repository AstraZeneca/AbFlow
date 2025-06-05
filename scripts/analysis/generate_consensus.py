#!/usr/bin/env python3
"""
compute_sabdab_consensus_and_pssm.py

1) Loads your YAML config (e.g. config.yaml) to find:
     data_root   = datamodule.dataset.paths.data
     dataset_name = datamodule.dataset.name
2) Opens the LMDB at:
       <data_root>/<dataset_name>/abflow_processed_structures_<dataset_name>_v4.lmdb
   and the pickle list:
       <data_root>/<dataset_name>/entries_list.pkl
3) For each entry:
     - region_index[i] == 2 → heavy chain residue
     - region_index[i] == 3 → light kappa residue
     - region_index[i] == 4 → light lambda residue
     - res_type[i] in [0..19] is a numeric token for a 3‐letter AA
   Converts each token → one‐letter code (via Bio.PDB’s protein_letters_3to1)
   to form three raw AA sequences per entry (heavy, kappa, lambda).
4) Collects three lists: heavy_seqs, kappa_seqs, lambda_seqs (only non‐empty sequences).
5) Runs ANARCI (scheme="aho") on each list separately to get:
     - aligned_maps: a list of dicts mapping Aho_position_str → residue
     - all_positions: a sorted list of every Aho_position_str across all sequences
   Builds a consensus at each position (majority‐vote). 
6) Constructs a simple PSSM (frequencies at each Aho position over {A,C,D,…,Y,'-'}) for each chain type.
7) Writes:
     - consensus_sequences.csv  (2 columns: chain,consensus_sequence)
     - heavy_pssm.csv            (columns: position,A,C,D,…,Y,- ; rows: one per Aho position)
     - kappa_pssm.csv            (same format)
     - lambda_pssm.csv           (same format)
"""

import os
import argparse
import yaml
import lmdb
import pickle
import zlib
import pandas as pd
from enum import IntEnum
from collections import Counter, OrderedDict
from abflow.utils.arguments import get_arguments, get_config, print_config_summary

# ANARCI for Aho numbering
from anarci import run_anarci

# Biopython for 3→1 letter mapping
from Bio.PDB.Polypeptide import protein_letters_3to1

# -----------------------------------------------------------------------------
# 1) Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute heavy/kappa/lambda consensus and PSSM from SABDab LMDB"
    )
    p.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to your YAML config file (e.g. config.yaml)"
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2) Load YAML config
# -----------------------------------------------------------------------------
def load_yaml_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_paths_from_config(cfg: dict):
    """
    Extract data_root = datamodule.dataset.paths.data
            dataset_name = datamodule.dataset.name
    """
    try:
        data_root = cfg["datamodule"]["dataset"]["paths"]["data"]
        dataset_name = cfg["datamodule"]["dataset"]["name"]
    except KeyError as e:
        raise KeyError(f"Could not find required key in config: {e}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return data_root, dataset_name


# -----------------------------------------------------------------------------
# 3) Open LMDB & load entries_list.pkl
# -----------------------------------------------------------------------------
def open_lmdb_env(data_root: str, dataset_name: str) -> lmdb.Environment:
    lmdb_path = os.path.join(
        data_root,
        dataset_name,
        f"abflow_processed_structures_{dataset_name}_v4.lmdb"
    )
    if not os.path.isfile(lmdb_path):
        raise FileNotFoundError(f"LMDB file not found at: {lmdb_path}")
    env = lmdb.open(
        lmdb_path,
        map_size=250 * 1024**3,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
    )
    return env


def load_all_entry_ids(data_root: str, dataset_name: str) -> list:
    entries_pkl = os.path.join(data_root, dataset_name, "entries_list.pkl")
    if not os.path.isfile(entries_pkl):
        raise FileNotFoundError(f"entries_list.pkl not found at: {entries_pkl}")
    with open(entries_pkl, "rb") as f:
        all_entries = pickle.load(f)
    if not isinstance(all_entries, list):
        raise RuntimeError("Expected entries_list.pkl to be a list of dicts with an 'id' key.")
    return all_entries


# -----------------------------------------------------------------------------
# 4) Define AminoAcid3 → one‐letter mapping
# -----------------------------------------------------------------------------
class AminoAcid3(IntEnum):
    ALA = 0
    CYS = 1
    ASP = 2
    GLU = 3
    PHE = 4
    GLY = 5
    HIS = 6
    ILE = 7
    LYS = 8
    LEU = 9
    MET = 10
    ASN = 11
    PRO = 12
    GLN = 13
    ARG = 14
    SER = 15
    THR = 16
    VAL = 17
    TRP = 18
    TYR = 19


# Build a dict: 0→'A', 1→'C', 2→'D', … 19→'Y'
index_to_1letter = {}
for aa3 in AminoAcid3:
    # aa3.name is the 3-letter code, e.g. 'ALA'
    aa1 = protein_letters_3to1[aa3.name]  # e.g. 'A'
    index_to_1letter[aa3.value] = aa1

# -----------------------------------------------------------------------------
# 5) Collect raw sequences from LMDB
# -----------------------------------------------------------------------------
def collect_three_chain_types(env: lmdb.Environment, all_entries: list):
    """
    For each entry in entries_list:
      - fetch pickled data via LMDB txn.get(id)
      - decompress, unpickle → `data` (a dict)
      - data["res_type"] is a list of ints in [0..19]
      - data["region_index"] is a list of ints (2=heavy, 3=κ, 4=λ)
    Build three lists of one-letter AA strings:
      heavy_seqs  = [ seq_of_all residues where region_index==2 ]
      kappa_seqs  = [ seq_of_all residues where region_index==3 ]
      lambda_seqs = [ seq_of_all residues where region_index==4 ]
    """
    heavy_seqs = []
    kappa_seqs = []
    lambda_seqs = []
    missing_count = 0

    with env.begin() as txn:
        for entry in all_entries:
            if (entry is None) or ("id" not in entry):
                continue

            db_id = entry["id"]
            raw = txn.get(db_id.encode())
            if raw is None:
                missing_count += 1
                continue

            data = pickle.loads(zlib.decompress(raw))

            if not isinstance(data, dict):
                continue

            # Must have both lists res_type and region_index, same length
            if ("res_type" not in data or "region_index" not in data):
                continue
            res_type = data["res_type"]
            region_index = data["region_index"]

            print(res_type)
            print(region_index)
            exit()
            if (not isinstance(res_type, (list, tuple))
                or not isinstance(region_index, (list, tuple))
                or len(res_type) != len(region_index)
            ):
                continue

            # Build three one-letter sequences (possibly empty)
            heavy_chars = []
            kappa_chars = []
            lambda_chars = []
            for aa_token, r in zip(res_type, region_index):
                # aa_token should be an int in [0..19]
                if not (isinstance(aa_token, (int,)) and (0 <= aa_token <= 19)):
                    continue
                aa1 = index_to_1letter[aa_token]

                if r == 2:
                    heavy_chars.append(aa1)
                elif r == 3:
                    kappa_chars.append(aa1)
                elif r == 4:
                    lambda_chars.append(aa1)
                else:
                    # region_index may have other values (e.g. antigen or padding)
                    continue

            heavy_seq = "".join(heavy_chars)
            kappa_seq = "".join(kappa_chars)
            lambda_seq = "".join(lambda_chars)

            if heavy_seq:
                heavy_seqs.append(heavy_seq)
            if kappa_seq:
                kappa_seqs.append(kappa_seq)
            if lambda_seq:
                lambda_seqs.append(lambda_seq)

    print(f"Total entries missing in LMDB: {missing_count}")
    print(f"Collected {len(heavy_seqs)} heavy sequences, {len(kappa_seqs)} kappa sequences, {len(lambda_seqs)} lambda sequences.")
    return heavy_seqs, kappa_seqs, lambda_seqs


# -----------------------------------------------------------------------------
# 6) Aho numbering (ANARCI) & alignment → position dictionaries
# -----------------------------------------------------------------------------
def align_with_aho(all_seqs, chain_type="H"):
    """
    Given a list of raw AA sequences (strings),
    run ANARCI with scheme="aho" to get numbering.

    Returns:
      aligned_maps: a list of dicts (one per seq) mapping Aho_position_str -> residue
      sorted_positions: a sorted list of all Aho_position_str observed
    """
    # Build input for ANARCI: [ ("H0", seq0), ("H1", seq1), … ]
    seqs_for_anarci = [(f"{chain_type}{i}", seq) for i, seq in enumerate(all_seqs)]
    anarci_results, _error = run_anarci(seqs_for_anarci, scheme="aho")

    aligned_maps = []
    all_positions = set()

    for result in anarci_results:
        # result is a list of domains; usually exactly one domain per chain
        if not result:
            # ANARCI failed → no numbering
            aligned_maps.append({})
            continue

        domain_numbering = result[0]  # pick first (only) domain
        this_map = {}
        for (_ch, pos_int, ins_code, aa) in domain_numbering:
            pos_label = str(pos_int) + ins_code  # e.g. "26", "52A", "95A"
            this_map[pos_label] = aa
            all_positions.add(pos_label)
        aligned_maps.append(this_map)

    # Sort by numeric part then insertion code
    def sort_key(label: str):
        num_part = ""
        letter_part = ""
        for c in label:
            if c.isdigit():
                num_part += c
            else:
                letter_part += c
        return (int(num_part), letter_part)

    sorted_positions = sorted(all_positions, key=sort_key)
    return aligned_maps, sorted_positions


# -----------------------------------------------------------------------------
# 7) Build consensus string (majority vote) for each position
# -----------------------------------------------------------------------------
def build_consensus(aligned_maps, all_positions):
    """
    Given:
      aligned_maps: [ {pos_label: residue, …}, … ]
      all_positions: [ pos_label1, pos_label2, … ] (sorted)
    Return one consensus string (length = len(all_positions))
    by majority‐voting at each position (treat missing as '-').
    """
    consensus_chars = []
    for pos in all_positions:
        counts = Counter()
        for amap in aligned_maps:
            aa = amap.get(pos, "-")
            counts[aa] += 1
        majority_aa, _ = counts.most_common(1)[0]
        consensus_chars.append(majority_aa)
    return "".join(consensus_chars)


# -----------------------------------------------------------------------------
# 8) Build a simple frequency‐based PSSM (positions × {A,..,Y,'-'})
# -----------------------------------------------------------------------------
def build_pssm(aligned_maps, all_positions):
    """
    Given aligned_maps and sorted_positions,
    produce an OrderedDict mapping each pos_label → dict of frequencies
       { 'A':0.07, 'C':0.00, …, 'Y':0.01, '-':0.12 }

    We count how many times each residue (including '-') appears at that position
    across aligned_maps, then divide by N_sequences.  Return as an OrderedDict
    keyed by pos_label.
    """
    num_seqs = len(aligned_maps)
    aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY-")  # we include '-' for gaps

    pssm = OrderedDict()
    for pos in all_positions:
        counts = Counter()
        for amap in aligned_maps:
            aa = amap.get(pos, "-")
            counts[aa] += 1
        freqs = {aa: (counts.get(aa, 0) / num_seqs) for aa in aa_alphabet}
        pssm[pos] = freqs
    return pssm


# -----------------------------------------------------------------------------
# 9) Main orchestration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    cfg = get_config(args)

    # 2) Extract data_root and dataset_name
    data_root, dataset_name = extract_paths_from_config(cfg)
    print(f"  → data_root   = {data_root}")
    print(f"  → dataset_name = {dataset_name}")

    # 3) Load entries_list.pkl
    print("Loading entries_list.pkl …")
    all_entries = load_all_entry_ids(data_root, dataset_name)
    print(f"  → Found {len(all_entries)} total entries in entries_list.pkl")

    # 4) Open LMDB
    print("Opening LMDB environment …")
    env = open_lmdb_env(data_root, dataset_name)

    # 5) Collect heavy / kappa / lambda raw sequences
    print("Collecting heavy, kappa, and lambda AA sequences …")
    heavy_seqs, kappa_seqs, lambda_seqs = collect_three_chain_types(env, all_entries)
    if len(heavy_seqs) == 0:
        raise RuntimeError("No heavy sequences found—check your LMDB and entries_list.pkl.")
    if len(kappa_seqs) == 0:
        print("Warning: found zero kappa sequences.")
    if len(lambda_seqs) == 0:
        print("Warning: found zero lambda sequences.")

    # 6) Aho numbering + alignment for heavy
    print("Running ANARCI (Aho numbering) on heavy sequences …")
    heavy_aligned_maps, heavy_positions = align_with_aho(heavy_seqs, chain_type="H")
    print(f"  → {len(heavy_aligned_maps)} heavy chains → {len(heavy_positions)} total Aho positions")

    # 7) Aho numbering + alignment for kappa
    if kappa_seqs:
        print("Running ANARCI (Aho numbering) on kappa sequences …")
        kappa_aligned_maps, kappa_positions = align_with_aho(kappa_seqs, chain_type="L")
        print(f"  → {len(kappa_aligned_maps)} kappa chains → {len(kappa_positions)} total Aho positions")
    else:
        kappa_aligned_maps, kappa_positions = [], []

    # 8) Aho numbering + alignment for lambda
    if lambda_seqs:
        print("Running ANARCI (Aho numbering) on lambda sequences …")
        lambda_aligned_maps, lambda_positions = align_with_aho(lambda_seqs, chain_type="L")
        print(f"  → {len(lambda_aligned_maps)} lambda chains → {len(lambda_positions)} total Aho positions")
    else:
        lambda_aligned_maps, lambda_positions = [], []

    # 9) Build consensus sequences
    print("Building consensus for heavy chains …")
    heavy_consensus = build_consensus(heavy_aligned_maps, heavy_positions)

    print("Building consensus for kappa chains …")
    if kappa_aligned_maps:
        kappa_consensus = build_consensus(kappa_aligned_maps, kappa_positions)
    else:
        kappa_consensus = ""

    print("Building consensus for lambda chains …")
    if lambda_aligned_maps:
        lambda_consensus = build_consensus(lambda_aligned_maps, lambda_positions)
    else:
        lambda_consensus = ""

    # 10) Save consensus sequences to CSV
    print("Saving consensus sequences to CSV …")
    df_cons = pd.DataFrame({
        "chain": ["heavy", "kappa", "lambda"],
        "consensus_sequence": [heavy_consensus, kappa_consensus, lambda_consensus]
    })
    out_cons_csv = "consensus_sequences.csv"
    df_cons.to_csv(out_cons_csv, index=False)
    print("  → Wrote consensus_sequences.csv at:", os.path.abspath(out_cons_csv))

    # 11) Build and save PSSM for heavy
    print("Building PSSM for heavy …")
    heavy_pssm = build_pssm(heavy_aligned_maps, heavy_positions)
    df_heavy_pssm = pd.DataFrame.from_dict(heavy_pssm, orient="index")
    df_heavy_pssm.index.name = "position"
    out_heavy_pssm = "heavy_pssm.csv"
    df_heavy_pssm.to_csv(out_heavy_pssm)
    print("  → Wrote heavy_pssm.csv at:", os.path.abspath(out_heavy_pssm))

    # 12) Build and save PSSM for kappa (if any)
    if kappa_aligned_maps:
        print("Building PSSM for kappa …")
        kappa_pssm = build_pssm(kappa_aligned_maps, kappa_positions)
        df_kappa_pssm = pd.DataFrame.from_dict(kappa_pssm, orient="index")
        df_kappa_pssm.index.name = "position"
        out_kappa_pssm = "kappa_pssm.csv"
        df_kappa_pssm.to_csv(out_kappa_pssm)
        print("  → Wrote kappa_pssm.csv at:", os.path.abspath(out_kappa_pssm))
    else:
        print("Skipping kappa PSSM (no kappa sequences).")

    # 13) Build and save PSSM for lambda (if any)
    if lambda_aligned_maps:
        print("Building PSSM for lambda …")
        lambda_pssm = build_pssm(lambda_aligned_maps, lambda_positions)
        df_lambda_pssm = pd.DataFrame.from_dict(lambda_pssm, orient="index")
        df_lambda_pssm.index.name = "position"
        out_lambda_pssm = "lambda_pssm.csv"
        df_lambda_pssm.to_csv(out_lambda_pssm)
        print("  → Wrote lambda_pssm.csv at:", os.path.abspath(out_lambda_pssm))
    else:
        print("Skipping lambda PSSM (no lambda sequences).")

    print("All done.")

