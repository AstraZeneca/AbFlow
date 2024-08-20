import os
import subprocess

pdb_dir = "./data/rabd/pdb/"
diffab_repo_dir = "./benchmarks/diffab/"
config_files = [
    "configs/test/codesign_single.yml",
    "configs/test/fixbb.yml",
    "configs/test/strpred.yml",
]
results_dirs = [
    "results/codesign_single/",
    "results/fixbb/",
    "results/strpred/",
]


original_pdb_files = {f for f in os.listdir(pdb_dir) if f.endswith(".pdb")}
original_working_dir = os.getcwd()

try:
    os.chdir(diffab_repo_dir)

    for pdb_file in original_pdb_files:
        pdb_file_path = os.path.join(original_working_dir, pdb_dir, pdb_file)

        for config_file, results_dir in zip(config_files, results_dirs):

            # check if the results directory exists
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            result_file_names = {f.split(".")[0] for f in os.listdir(results_dir)}
            pdb_file_name = pdb_file.split(".")[0]

            if pdb_file_name in result_file_names:
                print(
                    f"Skipping {pdb_file}.pdb as it is already processed in {results_dir}"
                )
                continue

            command = [
                "python",
                "design_pdb.py",
                pdb_file_path,
                "--config",
                config_file,
            ]

            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)

finally:
    os.chdir(original_working_dir)

    current_pdb_files = {f for f in os.listdir(pdb_dir) if f.endswith(".pdb")}
    additional_files = current_pdb_files - original_pdb_files

    for additional_file in additional_files:
        os.remove(os.path.join(pdb_dir, additional_file))

    print("Finished processing all PDB files. Additional files have been removed.")
