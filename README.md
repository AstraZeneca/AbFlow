# AbFlow

Adaptive Multi-Task Training for Antibody Sequence-Structure Generation via Flow Matching

![Maturity-level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white)

AbFlow is a generative model for joint antibody sequence and structure design. It can sample CDR loop sequences and all-atom conformations, conditioned on structural or sequence context and achieves state-of-the-art results on in silico and experimental benchmarks.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AstraZeneca/AbFlow.git
cd AbFlow
```

### 2. Create the conda environment

To create the environment:

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate abflow
```

### 4. Install the `abflow` package

```bash
pip install ./
```

Now, you are ready to run AbFlow.

---

## Main Dependencies

AbFlow is built primarily on:
- **PyTorch (>=2.4)**
- **Lightning (==2.2.1)**
- **BioPython (==1.84)**
- **pandas (==2.2.2)**
- **numpy (==1.26.4)**

See the [environment.yml](environment.yml) file for a complete list of packages and pinned versions.

---

## Usage

Below we show an example for redesigning CDR loops and generating a new PDB structure. Whether you redesign individual loops, all six loops or the entire antibody depends on your model checkpoint and its configuration.

For more details, use the `--help` option with any script.

### Example: Redesigning CDR Loops

```bash
python scripts/design/design_pdb.py \
    --config /path/to/config.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --input_pdb /path/to/antibody_complex.pdb \
    --heavy_chain H \
    --light_chain L \
    --antigen_chains A B \
    --scheme chothia \
    --output_dir results \
    --seed 2025 \
    --device cuda:0
```

- `--config`: Path to the configuration file associated with the model checkpoint.
- `--checkpoint`: Path to your trained model checkpoint.
- `--input_pdb`: Path to the input PDB file of the antibody complex.
- `--heavy_chain` and `--light_chain`: Chain identifiers for the antibody.
- `--antigen_chains`: Chain identifiers for any antigens present (optional).
- `--scheme`: Numbering scheme (commonly "chothia").
- `--output_dir`: Directory to store results.
- `--seed`: Random seed for reproducibility.
- `--device`: Device to run the computations on (e.g., `cuda:0` or `cpu`).

The output directory will include the redesigned PDB file.

---

## Contact

For any issues, suggestions, or questions, please reach out to:
- [hz362@cam.ac.uk](mailto:hz362@cam.ac.uk)
- [ucabtuc@gmail.com](mailto:ucabtuc@gmail.com)

---

## Reference

If you find this code useful, please cite:

```bibtex
@misc{our_paper,
  title={AbFlow: Adaptive Multi-Task Training for Antibody Sequence-Structure Generation},
  author={...},
  year={...},
  archivePrefix={...},
  primaryClass={...}
}
```