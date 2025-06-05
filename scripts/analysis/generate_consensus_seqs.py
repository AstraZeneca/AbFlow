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
3) Scans the entire LMDB once (single‐threaded) with a progress bar. For each entry:
     - chain_type[i] == 2 → heavy residue
     - chain_type[i] == 3 → kappa residue
     - chain_type[i] == 4 → lambda residue
     - res_type[i] in [0..19] is a numeric token for a 3‐letter AA  
   Converts token→one‐letter code via a prebuilt NumPy array. Builds three raw
   AA‐string lists: `heavy_seqs`, `kappa_seqs`, `lambda_seqs`.
4) Splits each list into `n_aho_workers` chunks; runs ANARCI (scheme="aho", 
   `ncpu=threads_per_worker`) on each chunk **via joblib**, showing a progress bar
   over chunks. Collects all partial `numbered` results and merges them into
   combined `aligned_maps` + `all_positions` per chain type.
5) Builds consensus strings + PSSMs in one NumPy pass per chain type.
6) Writes out:
     - consensus_sequences.csv
     - heavy_pssm.csv
     - kappa_pssm.csv
     - lambda_pssm.csv
"""

import os
import yaml
import lmdb
import pickle
import zlib
import pandas as pd
import numpy as np
import torch
import math
import multiprocessing as mp
from enum import IntEnum
from collections import OrderedDict
from tqdm import tqdm
from anarci import run_anarci
from Bio.PDB.Polypeptide import protein_letters_3to1
from joblib import Parallel, delayed
from abflow.utils.arguments import get_arguments, get_config

# -----------------------------------------------------------------------------
# 1) Define mapping from 3-letter tokens (0..19) → one-letter amino acids
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

# Prebuilt NumPy array: token (0..19) → one-letter code
letter_map = np.array(
    [protein_letters_3to1[aa3.name] for aa3 in AminoAcid3],
    dtype="<U1"
)

# -----------------------------------------------------------------------------
# 2) Load YAML config and extract paths
# -----------------------------------------------------------------------------
def load_yaml_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def extract_paths_from_config(cfg: dict):
    try:
        data_root = cfg["datamodule"]["dataset"]["paths"]["data"]
        dataset_name = cfg["datamodule"]["dataset"]["name"]
    except KeyError as e:
        raise KeyError(f"Missing key in config: {e}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return data_root, dataset_name

# -----------------------------------------------------------------------------
# 3) Open LMDB and load entry list
# -----------------------------------------------------------------------------
def open_lmdb_env(data_root: str, dataset_name: str) -> lmdb.Environment:
    lmdb_path = os.path.join(
        data_root,
        dataset_name,
        f"abflow_processed_structures_{dataset_name}_v4.lmdb"
    )
    if not os.path.isfile(lmdb_path):
        raise FileNotFoundError(f"LMDB file not found at: {lmdb_path}")
    return lmdb.open(
        lmdb_path,
        map_size=250 * 1024**3,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
    )

def load_all_entry_ids(data_root: str, dataset_name: str) -> list:
    entries_pkl = os.path.join(data_root, dataset_name, "entries_list.pkl")
    if not os.path.isfile(entries_pkl):
        raise FileNotFoundError(f"entries_list.pkl not found at: {entries_pkl}")
    with open(entries_pkl, "rb") as f:
        all_entries = pickle.load(f)
    if not isinstance(all_entries, list):
        raise RuntimeError("entries_list.pkl must be a list of dicts with an 'id' key.")
    return all_entries

# -----------------------------------------------------------------------------
# 4) Single‐threaded scan of LMDB (with tqdm) to collect raw sequences
# -----------------------------------------------------------------------------
def collect_three_chain_types_fast(env: lmdb.Environment, needed_ids: set):
    """
    Scans the LMDB exactly once (single‐threaded). For each (key, raw_value):
      - decode key → id_str
      - if id_str ∈ needed_ids, decompress/unpickle → data dict
      - data["res_type"], data["chain_type"] → two lists or two torch.Tensors
      - convert to NumPy arrays (int8), map tokens→letters in one shot, apply boolean masks
      - append heavy/kappa/lambda subsequences to three Python lists.
    Uses tqdm to show progress over total entries in the LMDB.

    Returns (heavy_seqs, kappa_seqs, lambda_seqs).
    """
    heavy_seqs = []
    kappa_seqs = []
    lambda_seqs = []

    # Determine total number of keys for the progress bar
    stat = env.stat()
    total_entries = stat.get("entries", 0)

    found = 0
    missing_ids = set(needed_ids)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, raw in tqdm(cursor, total=total_entries, desc="Scanning LMDB"):
            id_str = key.decode()
            if id_str in needed_ids:
                data = pickle.loads(zlib.decompress(raw))
                if isinstance(data, dict) and "res_type" in data and "chain_type" in data:
                    res_type = data["res_type"]
                    chain_type = data["chain_type"]

                    if (
                        (isinstance(res_type, (list, tuple)) or torch.is_tensor(res_type))
                        and (isinstance(chain_type, (list, tuple)) or torch.is_tensor(chain_type))
                        and len(res_type) == len(chain_type)
                    ):
                        # Convert to NumPy arrays
                        if torch.is_tensor(res_type):
                            rt_arr = res_type.cpu().numpy().astype(np.int8)
                        else:
                            rt_arr = np.asarray(res_type, dtype=np.int8)

                        if torch.is_tensor(chain_type):
                            ct_arr = chain_type.cpu().numpy().astype(np.int8)
                        else:
                            ct_arr = np.asarray(chain_type, dtype=np.int8)

                        # Map tokens → letters
                        letters = letter_map[rt_arr]  # shape (L,), dtype "<U1"

                        # Boolean masks
                        mask_heavy  = (ct_arr == 2)
                        mask_kappa  = (ct_arr == 3)
                        mask_lambda = (ct_arr == 4)

                        if mask_heavy.any():
                            heavy_seqs.append("".join(letters[mask_heavy].tolist()))
                        if mask_kappa.any():
                            kappa_seqs.append("".join(letters[mask_kappa].tolist()))
                        if mask_lambda.any():
                            lambda_seqs.append("".join(letters[mask_lambda].tolist()))

                found += 1
                missing_ids.discard(id_str)
                if found >= len(needed_ids):
                    break

    print(f"  → Found {found}/{len(needed_ids)} requested entries. Missing: {len(missing_ids)}.")
    print(f"  → Collected {len(heavy_seqs)} heavy, {len(kappa_seqs)} kappa, {len(lambda_seqs)} lambda sequences.")
    return heavy_seqs, kappa_seqs, lambda_seqs

# -----------------------------------------------------------------------------
# 5) Helper to run ANARCI on a chunk of sequences
# -----------------------------------------------------------------------------
def _anarci_chunk(seqs_with_names, scheme, ncpu):
    """
    seqs_with_names: list of (name, sequence) pairs, where 'name' is string ID
    scheme="aho"
    ncpu = # of threads passed to run_anarci
    Returns: the `numbered` list (one element per sequence in seqs_with_names).
    """
    # run_anarci returns (sequences_out, numbered, alignment_details, hit_tables)
    _, numbered, _, _ = run_anarci(seqs_with_names, scheme=scheme, ncpu=ncpu)
    return numbered

def align_with_aho_in_parallel(raw_seqs, chain_prefix, n_aho_workers=None, threads_per_worker=None):
    """
    Splits raw_seqs (list of AA strings) into n_aho_workers chunks. Each chunk is processed
    by joblib.Parallel calling run_anarci(..., scheme="aho", ncpu=threads_per_worker).
    Shows a tqdm over the chunks.

    Returns (aligned_maps, sorted_positions).
    """
    total_seqs = len(raw_seqs)
    if total_seqs == 0:
        return [], []

    n_cores = mp.cpu_count()
    if n_aho_workers is None:
        n_aho_workers = min(n_cores, total_seqs)
    if threads_per_worker is None:
        threads_per_worker = max(1, n_cores // n_aho_workers)

    chunk_size = math.ceil(total_seqs / n_aho_workers)
    idx_chunks = [list(range(i, min(total_seqs, i + chunk_size))) for i in range(0, total_seqs, chunk_size)]

    # Build argument list for each chunk
    args_list = []
    for chunk_indices in idx_chunks:
        seqs_with_names = [(f"{chain_prefix}{i}", raw_seqs[i]) for i in chunk_indices]
        args_list.append((seqs_with_names, "aho", threads_per_worker))

    # Run joblib.Parallel over chunks, show progress
    numbered_chunks = Parallel(
        n_jobs=len(args_list),
        backend="loky",
        verbose=5
    )(
        delayed(_anarci_chunk)(seqs_with_names, scheme, ncpu)
        for (seqs_with_names, scheme, ncpu) in args_list
    )

    # Flatten numbered_chunks (list of lists) into one list in original order
    numbered_all = []
    for numbered_chunk in numbered_chunks:
        numbered_all.extend(numbered_chunk)

    aligned_maps = []
    all_positions = set()
    for entry in numbered_all:
        if entry is None:
            aligned_maps.append({})
            continue
        first_domain = entry[0]  # ( [((pos_int, ins_code), aa), ...], start, end )
        domain_numbering = first_domain[0]
        amap = {}
        for (pos_int, ins_code), aa in domain_numbering:
            pos_label = str(pos_int) + ins_code
            amap[pos_label] = aa
            all_positions.add(pos_label)
        aligned_maps.append(amap)

    def sort_key(label: str):
        num_part = ""
        ins_part = ""
        for c in label:
            if c.isdigit():
                num_part += c
            else:
                ins_part += c
        return (int(num_part), ins_part)

    sorted_positions = sorted(all_positions, key=sort_key)
    return aligned_maps, sorted_positions

# -----------------------------------------------------------------------------
# 6) Build consensus + PSSM
# -----------------------------------------------------------------------------
def build_consensus_and_pssm(aligned_maps, all_positions):
    """
    aligned_maps: [ {pos_label: aa}, ... ], length = n_seq
    all_positions: sorted list of pos_label, length = n_pos
    Returns (consensus_str, df_pssm).
    """
    n_seq = len(aligned_maps)
    n_pos = len(all_positions)
    pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}

    mat = np.full((n_seq, n_pos), "-", dtype="<U1")
    for i, amap in enumerate(aligned_maps):
        for pos, aa in amap.items():
            mat[i, pos_to_idx[pos]] = aa

    aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY-")
    consensus_chars = []
    freq_dicts = OrderedDict()

    for col_idx, pos_label in enumerate(all_positions):
        col = mat[:, col_idx]
        uniques, counts = np.unique(col, return_counts=True)
        majority_aa = uniques[np.argmax(counts)]
        consensus_chars.append(majority_aa)
        count_map = dict(zip(uniques, counts))
        freqs = {aa: (count_map.get(aa, 0) / n_seq) for aa in aa_alphabet}
        freq_dicts[pos_label] = freqs

    consensus_str = "".join(consensus_chars)
    df_pssm = pd.DataFrame.from_dict(freq_dicts, orient="index")
    df_pssm.index.name = "position"
    return consensus_str, df_pssm

# -----------------------------------------------------------------------------
# 7) Main orchestration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_arguments()
    cfg = get_config(args)

    data_root, dataset_name = extract_paths_from_config(cfg)
    print(f"  → data_root   = {data_root}")
    print(f"  → dataset_name = {dataset_name}")

    # 7.1) Load entry IDs
    print("Loading entries_list.pkl …")
    all_entries = load_all_entry_ids(data_root, dataset_name)
    print(f"  → Found {len(all_entries)} total entries.")

    # 7.2) Open LMDB (readonly)
    print("Opening LMDB environment …")
    env = open_lmdb_env(data_root, dataset_name)

    # 7.3) Build set of needed IDs
    needed_ids = {e["id"] for e in all_entries if isinstance(e, dict) and "id" in e}

    # 7.4) Single‐threaded scan to collect raw sequences
    print("Collecting heavy/kappa/lambda AA sequences …")
    heavy_seqs, kappa_seqs, lambda_seqs = collect_three_chain_types_fast(env, needed_ids)

    if not heavy_seqs:
        raise RuntimeError("No heavy sequences found—check your LMDB.")
    if not kappa_seqs:
        print("Warning: found zero kappa sequences.")
    if not lambda_seqs:
        print("Warning: found zero lambda sequences.")

    # 7.5) Parallel ANARCI on heavy
    n_cores = mp.cpu_count()
    print("Running ANARCI (aho) on heavy sequences …")
    heavy_aligned_maps, heavy_positions = align_with_aho_in_parallel(
        heavy_seqs,
        chain_prefix="H",
        n_aho_workers=min(n_cores, len(heavy_seqs)),
        threads_per_worker=max(1, n_cores // min(n_cores, len(heavy_seqs)))
    )
    print(f"  → {len(heavy_aligned_maps)} heavy chains → {len(heavy_positions)} positions.")

    # 7.6) Parallel ANARCI on kappa
    if kappa_seqs:
        print("Running ANARCI (aho) on kappa sequences …")
        kappa_aligned_maps, kappa_positions = align_with_aho_in_parallel(
            kappa_seqs,
            chain_prefix="L",
            n_aho_workers=min(n_cores, len(kappa_seqs)),
            threads_per_worker=max(1, n_cores // min(n_cores, len(kappa_seqs)))
        )
        print(f"  → {len(kappa_aligned_maps)} kappa chains → {len(kappa_positions)} positions.")
    else:
        kappa_aligned_maps, kappa_positions = [], []

    # 7.7) Parallel ANARCI on lambda
    if lambda_seqs:
        print("Running ANARCI (aho) on lambda sequences …")
        lambda_aligned_maps, lambda_positions = align_with_aho_in_parallel(
            lambda_seqs,
            chain_prefix="L",
            n_aho_workers=min(n_cores, len(lambda_seqs)),
            threads_per_worker=max(1, n_cores // min(n_cores, len(lambda_seqs)))
        )
        print(f"  → {len(lambda_aligned_maps)} lambda chains → {len(lambda_positions)} positions.")
    else:
        lambda_aligned_maps, lambda_positions = [], []

    # 7.8) Build consensus + PSSM
    print("Building consensus and PSSM for heavy chains …")
    heavy_consensus, df_heavy_pssm = build_consensus_and_pssm(heavy_aligned_maps, heavy_positions)

    if kappa_aligned_maps:
        print("Building consensus and PSSM for kappa chains …")
        kappa_consensus, df_kappa_pssm = build_consensus_and_pssm(kappa_aligned_maps, kappa_positions)
    else:
        kappa_consensus, df_kappa_pssm = "", None

    if lambda_aligned_maps:
        print("Building consensus and PSSM for lambda chains …")
        lambda_consensus, df_lambda_pssm = build_consensus_and_pssm(lambda_aligned_maps, lambda_positions)
    else:
        lambda_consensus, df_lambda_pssm = "", None

    # 7.9) Write CSVs
    print("Saving consensus sequences to CSV …")
    df_cons = pd.DataFrame({
        "chain": ["heavy", "kappa", "lambda"],
        "consensus_sequence": [heavy_consensus, kappa_consensus, lambda_consensus]
    })
    df_cons.to_csv("consensus_sequences.csv", index=False)
    print("  → Wrote consensus_sequences.csv")

    print("Saving heavy PSSM …")
    df_heavy_pssm.to_csv("heavy_pssm.csv")
    print("  → Wrote heavy_pssm.csv")

    if df_kappa_pssm is not None:
        print("Saving kappa PSSM …")
        df_kappa_pssm.to_csv("kappa_pssm.csv")
        print("  → Wrote kappa_pssm.csv")
    else:
        print("Skipping kappa PSSM (none).")

    if df_lambda_pssm is not None:
        print("Saving lambda PSSM …")
        df_lambda_pssm.to_csv("lambda_pssm.csv")
        print("  → Wrote lambda_pssm.csv")
    else:
        print("Skipping lambda PSSM (none).")

    print("All done.")
