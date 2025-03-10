"""
This script process a saved LMDB database to the input data dict in lmdb format for abflow.

A saved LMDB database contains the following files:
- entries_list.pkl: A list of dictionaries containing the entry ID for each complex.
- structures.lmdb: The LMDB database containing the structure data for each complex.

To use this script, run: python scripts/data/process_lmdb.py --path <path_to_data_folder>
Example command to process sabdab data: python abflow/scripts/data/process_lmdb.py --path /scratch/hz362/datavol/data/sabdab
Note: If you want to run clustering on top of it, use --cluster command.
"""
import argparse
import os
import pickle
import zlib
import lmdb
import math
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from abflow.data.process_pdb import process_lmdb_chain, add_features
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def read_valid_entries(data_folder):
    entries_path = os.path.join(data_folder, "entries_list.pkl")
    with open(entries_path, "rb") as f:
        all_entries = pickle.load(f)
    return [e for e in all_entries if e and "id" in e]

def process_single_entry(args):
    db_id, structure_data = args
    data = pickle.loads(structure_data)
    processed_data = process_lmdb_chain(data)
    processed_data.update(add_features(processed_data))
    return db_id, zlib.compress(pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL))

def process_lmdb(data_folder: str):
    """
    Process the LMDB database to the input data dict for abflow.
    """
    dataset_name = '_'.join(os.path.basename(os.path.normpath(data_folder)).split('_')[:-1])
    source_db_path = os.path.join(data_folder, "structures.lmdb")
    output_db_path = f"/home/jovyan/abflow-datavol/data/{dataset_name}/abflow_processed_structures_{dataset_name}_v3.lmdb"
    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    valid_entries = read_valid_entries(data_folder)
    if not valid_entries:
        print("No valid entries found. Exiting.")
        return

    if dataset_name == "sabdab":
        map_size = 100 * 1024**3  # 100 GB
    else:
        map_size = 2000 * 1024**3  # 2 TB

    source_db = lmdb.open(
        source_db_path,
        map_size=map_size,
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    output_db = lmdb.open(output_db_path, map_size=map_size, subdir=False)

    # Pick workers
    cpu_count = os.cpu_count() or 1
    max_workers = min(4, cpu_count)

    # Pick a chunk size
    chunk_size = 75

    with ProcessPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(valid_entries)) as pbar:
        for i in range(0, len(valid_entries), chunk_size):
            chunk = valid_entries[i : i + chunk_size]
            # Read LMDB data in a single transaction for the chunk
            chunk_data = []
            with source_db.begin() as txn:
                for entry in chunk:
                    db_id = entry["id"]
                    structure_data = txn.get(db_id.encode())
                    if structure_data is not None:
                        chunk_data.append((db_id, structure_data))
            if not chunk_data:
                pbar.update(len(chunk))
                continue

            # Parallel process
            results = list(executor.map(process_single_entry, chunk_data))
            # Write to new DB in one transaction
            with output_db.begin(write=True) as wtxn:
                for db_id, compressed_data in results:
                    wtxn.put(db_id.encode(), compressed_data)
            pbar.update(len(chunk))

    source_db.close()
    output_db.close()
    print("Preprocessing complete. Data saved to:", output_db_path)
    return output_db_path

def generate_clusters(processed_db_path, processed_dir):
    """
    Generate clusters for antibodies using mmseqs2.
    
    This function opens the processed LMDB database, extracts the CDR3 sequences
    (using the heavy chain H3_seq if available, else the light chain L3_seq), writes them
    to a FASTA file, and then runs mmseqs2 clustering.
    """
    cdr_records = []
    # Open the processed LMDB database in read-only mode; note subdir=False to match how it was created.
    env = lmdb.open(processed_db_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            try:
                structure = pickle.loads(zlib.decompress(val))
            except Exception as e:
                print(f"Could not load structure {key}: {e}")
                continue
            # Use the key as the structure id
            structure_id = key.decode() if isinstance(key, bytes) else str(key)
            # Prefer heavy chain clustering (H3_seq) over light chain (L3_seq)
            if "H3_seq" in structure and structure["H3_seq"] is not None:
                seq = structure["H3_seq"]
                cdr_records.append(SeqRecord(Seq(seq),
                                             id=structure_id,
                                             name = '',
                                             description=""))
            elif "L3_seq" in structure and structure["L3_seq"] is not None:
                seq = structure["L3_seq"]
                cdr_records.append(SeqRecord(Seq(seq),
                                             id=structure_id,
                                             name = '',
                                             description=""))
    env.close()

    # Write the FASTA file in the same directory as the processed database
    fasta_path = os.path.join(processed_dir, 'cdr_sequences.fasta')
    SeqIO.write(cdr_records, fasta_path, 'fasta')
            
    print("CDR sequences written to:", fasta_path)

    # Prepare command for clustering 
    cmd = ' '.join([
        'mmseqs', 
        'easy-cluster',
        os.path.realpath(fasta_path),
        'cluster_result_clustermode1', 
        'cluster_tmp',
        '--min-seq-id', '0.5',
        '-c', '0.8',
        '--cov-mode', '1',
    ])
    print("Running clustering command:")
    print(cmd)

    # Run the command in the processed directory
    subprocess.run(cmd, cwd=processed_dir, shell=True, check=True)
    print("Clustering complete. Results saved in:", processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a saved LMDB database.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the data folder containing 'entries_list.pkl' and 'structures.lmdb'.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="If set, run antibody clustering based on CDR3 sequences using mmseqs2.",
    )
    args = parser.parse_args()

    data_folder = args.path
    if not os.path.isdir(data_folder):
        raise ValueError(f"The provided path '{data_folder}' is not a valid directory.")

    # Some basic checks
    required_files = ["entries_list.pkl", "structures.lmdb"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_folder, f))]
    if missing_files:
        raise ValueError(f"The following required files are missing in '{data_folder}': {', '.join(missing_files)}")

    # Process the LMDB database and get the output database path
    processed_db_path = process_lmdb(data_folder)

    # If clustering is enabled, run the clustering function.
    if args.cluster and processed_db_path is not None:
        processed_dir = os.path.dirname(processed_db_path)
        generate_clusters(processed_db_path, processed_dir)