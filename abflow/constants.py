"""
Contains constants used in protein structure analysis.
"""

import torch
from enum import Enum, IntEnum
from Bio.PDB.Polypeptide import protein_letters_3to1

from .geometry import create_rotation_matrix

# Chain and region mappings
chain_id_to_index = {
    "antigen": 0,
    "heavy": 1,
    "light_kappa": 2,
    "light_lambda": 3,
}
antibody_chain_names = ["heavy", "light_kappa", "light_lambda"]
antigen_chain_name = ["antigen"]
antibody_index = [chain_id_to_index[chain] for chain in antibody_chain_names]
antigen_index = [chain_id_to_index[chain] for chain in antigen_chain_name]

region_to_index = {
    "framework": 0,
    "hcdr1": 1,
    "hcdr2": 2,
    "hcdr3": 3,
    "lcdr1": 4,
    "lcdr2": 5,
    "lcdr3": 6,
    "antigen": 7,
}


# Stores indices under the 3-letter code of each AA
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


AminoAcid3Index = {aa: index for index, aa in AminoAcid3.__members__.items()}

# Stores indices under the 1-letter code of each AA
AminoAcid1 = IntEnum(
    "AminoAcid1",
    {protein_letters_3to1[aa]: index for aa, index in AminoAcid3.__members__.items()},
)

AminoAcid1Index = {aa: index for index, aa in AminoAcid1.__members__.items()}


# Mappings between amino acid indices and their 3-letter and 1-letter codes
aa3_index_to_name = {v: k for k, v in AminoAcid3.__members__.items()}
aa1_index_to_name = {v: k for k, v in AminoAcid1.__members__.items()}
aa3_name_to_index = {k: v for k, v in AminoAcid3.__members__.items()}
aa1_name_to_index = {k: v for k, v in AminoAcid1.__members__.items()}


# fmt: off
chi_angles_atoms = {
    AminoAcid3.ALA: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AminoAcid3.ARG: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    AminoAcid3.ASN: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AminoAcid3.ASP: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AminoAcid3.CYS: [["N", "CA", "CB", "SG"]],
    AminoAcid3.GLN: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    AminoAcid3.GLU: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    AminoAcid3.GLY: [],
    AminoAcid3.HIS: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    AminoAcid3.ILE: [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    AminoAcid3.LEU: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.LYS: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    AminoAcid3.MET: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    AminoAcid3.PHE: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.PRO: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    AminoAcid3.SER: [["N", "CA", "CB", "OG"]],
    AminoAcid3.THR: [["N", "CA", "CB", "OG1"]],
    AminoAcid3.TRP: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.TYR: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.VAL: [["N", "CA", "CB", "CG1"]],
}


chi_angles_mask = {
    AminoAcid3.ALA: [False, False, False, False],
    AminoAcid3.ARG: [True, True, True, True],
    AminoAcid3.ASN: [True, True, False, False],
    AminoAcid3.ASP: [True, True, False, False],
    AminoAcid3.CYS: [True, False, False, False],
    AminoAcid3.GLN: [True, True, True, False],
    AminoAcid3.GLU: [True, True, True, False],
    AminoAcid3.GLY: [False, False, False, False],
    AminoAcid3.HIS: [True, True, False, False],
    AminoAcid3.ILE: [True, True, False, False],
    AminoAcid3.LEU: [True, True, False, False],
    AminoAcid3.LYS: [True, True, True, True],
    AminoAcid3.MET: [True, True, True, False],
    AminoAcid3.PHE: [True, True, False, False],
    AminoAcid3.PRO: [True, True, False, False],
    AminoAcid3.SER: [True, False, False, False],
    AminoAcid3.THR: [True, False, False, False],
    AminoAcid3.TRP: [True, True, False, False],
    AminoAcid3.TYR: [True, True, False, False],
    AminoAcid3.VAL: [True, False, False, False],
}

restype_to_heavyatom_names = {
    AminoAcid3.ALA: ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.ARG: ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", "", "OXT"],
    AminoAcid3.ASN: ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", "", "OXT"],
    AminoAcid3.ASP: ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", "", "OXT"],
    AminoAcid3.CYS: ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.GLN: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", "", "OXT"],
    AminoAcid3.GLU: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", "", "OXT"],
    AminoAcid3.GLY: ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.HIS: ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", "", "OXT"],
    AminoAcid3.ILE: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", "", "OXT"],
    AminoAcid3.LEU: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", "", "OXT"],
    AminoAcid3.LYS: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", "", "OXT"],
    AminoAcid3.MET: ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", "", "OXT"],
    AminoAcid3.PHE: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", "", "OXT"],
    AminoAcid3.PRO: ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.SER: ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.THR: ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "", "OXT"],
    AminoAcid3.TRP: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "OXT"],
    AminoAcid3.TYR: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", "", "OXT"],
    AminoAcid3.VAL: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", "", "OXT"],
}
# fmt: on

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_to_heavyatom_names.items()
}


class Torsion(IntEnum):
    BACKBONE = 0
    CHI1 = 1
    CHI2 = 2
    CHI3 = 3
    CHI4 = 4
    PSI = 5  # for oxygen imputation


HEAVY_ATOM_COORDS = {
    AminoAcid3.ALA: [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 5, (0.627, 1.062, 0.000)],
    ],
    AminoAcid3.ARG: [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 5, (0.626, 1.062, 0.000)],
        ["CG", 1, (0.616, 1.390, -0.000)],
        ["CD", 2, (0.564, 1.414, 0.000)],
        ["NE", 3, (0.539, 1.357, -0.000)],
        ["NH1", 4, (0.206, 2.301, 0.000)],
        ["NH2", 4, (2.078, 0.978, -0.000)],
        ["CZ", 4, (0.758, 1.093, -0.000)],
    ],
    AminoAcid3.ASN: [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 5, (0.625, 1.062, 0.000)],
        ["CG", 1, (0.584, 1.399, 0.000)],
        ["ND2", 2, (0.593, -1.188, 0.001)],
        ["OD1", 2, (0.633, 1.059, 0.000)],
    ],
    AminoAcid3.ASP: [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 5, (0.626, 1.062, -0.000)],
        ["CG", 1, (0.593, 1.398, -0.000)],
        ["OD1", 2, (0.610, 1.091, 0.000)],
        ["OD2", 2, (0.592, -1.101, -0.003)],
    ],
    AminoAcid3.CYS: [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 5, (0.625, 1.062, -0.000)],
        ["SG", 1, (0.728, 1.653, 0.000)],
    ],
    AminoAcid3.GLN: [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 5, (0.626, 1.062, -0.000)],
        ["CG", 1, (0.615, 1.393, 0.000)],
        ["CD", 2, (0.587, 1.399, -0.000)],
        ["NE2", 3, (0.593, -1.189, -0.001)],
        ["OE1", 3, (0.634, 1.060, 0.000)],
    ],
    AminoAcid3.GLU: [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 5, (0.626, 1.062, 0.000)],
        ["CG", 1, (0.615, 1.392, 0.000)],
        ["CD", 2, (0.600, 1.397, 0.000)],
        ["OE1", 3, (0.607, 1.095, -0.000)],
        ["OE2", 3, (0.589, -1.104, -0.001)],
    ],
    AminoAcid3.GLY: [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 5, (0.626, 1.062, -0.000)],
    ],
    AminoAcid3.HIS: [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 5, (0.625, 1.063, 0.000)],
        ["CG", 1, (0.600, 1.370, -0.000)],
        ["CD2", 2, (0.889, -1.021, 0.003)],
        ["ND1", 2, (0.744, 1.160, -0.000)],
        ["CE1", 2, (2.030, 0.851, 0.002)],
        ["NE2", 2, (2.145, -0.466, 0.004)],
    ],
    AminoAcid3.ILE: [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 5, (0.627, 1.062, -0.000)],
        ["CG1", 1, (0.534, 1.437, -0.000)],
        ["CG2", 1, (0.540, -0.785, -1.199)],
        ["CD1", 2, (0.619, 1.391, 0.000)],
    ],
    AminoAcid3.LEU: [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 5, (0.625, 1.063, -0.000)],
        ["CG", 1, (0.678, 1.371, 0.000)],
        ["CD1", 2, (0.530, 1.430, -0.000)],
        ["CD2", 2, (0.535, -0.774, 1.200)],
    ],
    AminoAcid3.LYS: [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 5, (0.626, 1.062, -0.000)],
        ["CG", 1, (0.619, 1.390, 0.000)],
        ["CD", 2, (0.559, 1.417, 0.000)],
        ["CE", 3, (0.560, 1.416, 0.000)],
        ["NZ", 4, (0.554, 1.387, 0.000)],
    ],
    AminoAcid3.MET: [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 5, (0.625, 1.062, -0.000)],
        ["CG", 1, (0.613, 1.391, -0.000)],
        ["SD", 2, (0.703, 1.695, 0.000)],
        ["CE", 3, (0.320, 1.786, -0.000)],
    ],
    AminoAcid3.PHE: [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 5, (0.626, 1.062, -0.000)],
        ["CG", 1, (0.607, 1.377, 0.000)],
        ["CD1", 2, (0.709, 1.195, -0.000)],
        ["CD2", 2, (0.706, -1.196, 0.000)],
        ["CE1", 2, (2.102, 1.198, -0.000)],
        ["CE2", 2, (2.098, -1.201, -0.000)],
        ["CZ", 2, (2.794, -0.003, -0.001)],
    ],
    AminoAcid3.PRO: [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 5, (0.621, 1.066, 0.000)],
        ["CG", 1, (0.382, 1.445, 0.0)],
        # ['CD', 2, (0.427, 1.440, 0.0)],
        ["CD", 2, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    AminoAcid3.SER: [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 5, (0.626, 1.062, -0.000)],
        ["OG", 1, (0.503, 1.325, 0.000)],
    ],
    AminoAcid3.THR: [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 5, (0.626, 1.062, 0.000)],
        ["CG2", 1, (0.550, -0.718, -1.228)],
        ["OG1", 1, (0.472, 1.353, 0.000)],
    ],
    AminoAcid3.TRP: [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 5, (0.627, 1.062, 0.000)],
        ["CG", 1, (0.609, 1.370, -0.000)],
        ["CD1", 2, (0.824, 1.091, 0.000)],
        ["CD2", 2, (0.854, -1.148, -0.005)],
        ["CE2", 2, (2.186, -0.678, -0.007)],
        ["CE3", 2, (0.622, -2.530, -0.007)],
        ["NE1", 2, (2.140, 0.690, -0.004)],
        ["CH2", 2, (3.028, -2.890, -0.013)],
        ["CZ2", 2, (3.283, -1.543, -0.011)],
        ["CZ3", 2, (1.715, -3.389, -0.011)],
    ],
    AminoAcid3.TYR: [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 5, (0.627, 1.062, -0.000)],
        ["CG", 1, (0.607, 1.382, -0.000)],
        ["CD1", 2, (0.716, 1.195, -0.000)],
        ["CD2", 2, (0.713, -1.194, -0.001)],
        ["CE1", 2, (2.107, 1.200, -0.002)],
        ["CE2", 2, (2.104, -1.201, -0.003)],
        ["OH", 2, (4.168, -0.002, -0.005)],
        ["CZ", 2, (2.791, -0.001, -0.003)],
    ],
    AminoAcid3.VAL: [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 5, (0.627, 1.062, -0.000)],
        ["CG1", 1, (0.540, 1.429, -0.000)],
        ["CG2", 1, (0.533, -0.776, 1.203)],
    ],
}


PREVIOUS_SIDECHAIN_ATOM_ROTATIONS = torch.zeros([20, 6, 3, 3])
PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS = torch.zeros([20, 6, 3])
HEAVY_ATOM_TORSION_INDEX = torch.zeros([20, 15], dtype=torch.int64)
HEAYY_ATOM_POSITIONS = torch.zeros([20, 15, 3])


def _init_heavy_atom_constants():

    for aa in AminoAcid3:
        atom_groups = {name: group for name, group, _ in HEAVY_ATOM_COORDS[aa]}
        atom_positions = {
            name: torch.FloatTensor(pos) for name, _, pos in HEAVY_ATOM_COORDS[aa]
        }

        # heavy atom 15 positions
        for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[aa]):
            if (atom_name == "") or (atom_name not in atom_groups):
                continue
            HEAVY_ATOM_TORSION_INDEX[aa, atom_idx] = atom_groups[atom_name]
            HEAYY_ATOM_POSITIONS[aa, atom_idx, :] = atom_positions[atom_name]

        # idenity matrix for backbone
        PREVIOUS_SIDECHAIN_ATOM_ROTATIONS[aa, Torsion.BACKBONE] = torch.eye(3)
        PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS[aa, Torsion.BACKBONE] = torch.zeros(3)

        # psi torsion for oxygen imputation
        PREVIOUS_SIDECHAIN_ATOM_ROTATIONS[aa, Torsion.PSI] = create_rotation_matrix(
            v1=atom_positions["C"] - atom_positions["CA"],
            v2=atom_positions["N"] - atom_positions["CA"],
        )
        PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS[aa, Torsion.PSI] = atom_positions["C"]

        # previous sidechain atom rotations and translations
        # for chi1
        if chi_angles_mask[aa][0]:
            base_atom_name = chi_angles_atoms[aa][0]
            base_atom_position = [atom_positions[name] for name in base_atom_name]
            PREVIOUS_SIDECHAIN_ATOM_ROTATIONS[aa, Torsion.CHI1, :, :] = (
                create_rotation_matrix(
                    v1=base_atom_position[2] - base_atom_position[1],
                    v2=base_atom_position[0] - base_atom_position[1],
                )
            )
            PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS[aa, Torsion.CHI1, :] = (
                base_atom_position[2]
            )

        # for chi2, chi3, chi4
        for chi_idx in range(1, 4):
            if chi_angles_mask[aa][chi_idx]:
                previous_end_atom_name = chi_angles_atoms[aa][chi_idx][2]
                previous_end_atom_position = atom_positions[previous_end_atom_name]
                PREVIOUS_SIDECHAIN_ATOM_ROTATIONS[aa, chi_idx + Torsion.CHI1, :, :] = (
                    create_rotation_matrix(
                        v1=previous_end_atom_position, v2=torch.FloatTensor([-1, 0, 0])
                    )
                )
                PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS[aa, chi_idx + Torsion.CHI1, :] = (
                    previous_end_atom_position
                )


_init_heavy_atom_constants()


def get_heavy_atom_constants(
    res_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N_batch, N_res = res_type.size()
    res_type = res_type.flatten()
    prev_rotation = PREVIOUS_SIDECHAIN_ATOM_ROTATIONS.to(res_type.device)[
        res_type
    ].reshape(N_batch, N_res, 6, 3, 3)
    prev_translation = PREVIOUS_SIDECHAIN_ATOM_TRANSLATIONS.to(res_type.device)[
        res_type
    ].reshape(N_batch, N_res, 6, 3)
    torsion_index = HEAVY_ATOM_TORSION_INDEX.to(res_type.device)[res_type].reshape(
        N_batch, N_res, 15
    )
    atom14_position = HEAYY_ATOM_POSITIONS.to(res_type.device)[res_type].reshape(
        N_batch, N_res, 15, 3
    )
    return prev_rotation, prev_translation, torsion_index, atom14_position


def get_dihedral_mask(res_type: torch.Tensor) -> torch.Tensor:
    N_batch, N_res = res_type.size()
    res_type = res_type.flatten()
    mask = torch.tensor(
        [chi_angles_mask[aa] for aa in res_type.cpu().numpy()],
        dtype=torch.bool,
        device=res_type.device,
    ).reshape(N_batch, N_res, 4)
    return mask


class BackboneBondLengths(Enum):
    """
    Enum for storing mean backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 1.459
    CA_C = 1.525
    C_N = 1.336
    C_O = 1.229
    CA_CB = 1.532


class BackboneBondLengthStdDevs(Enum):
    """
    Enum for storing standard deviations in backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 0.020
    CA_C = 0.026
    C_N = 0.023


class BackboneBondAngles(Enum):
    """
    Enum for storing backbone bond angles, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 1.937
    CA_C_N = 2.046
    C_N_CA = 2.124


class BackboneBondAngleStdDevs(Enum):
    """
    Enum for storing backbone bond angle standard deviations, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 0.047
    CA_C_N = 0.038
    C_N_CA = 0.044


class AtomVanDerWaalRadii(Enum):
    """
    Enum for storing atomic van der waal radii, taken from table 1:

    Batsanov, S.S. Van der Waals Radii of Elements.
    Inorganic Materials 37, 871–885 (2001). https://doi.org/10.1023/A:1011625728803
    """

    C = 1.77
    N = 1.64
    O = 1.58
    S = 1.81


class Liability(Enum):
    """
    Enumeration for antibody liabilities within the CDR regions, with motifs and amino acid indices.

    Liability flags are taken from:
    Satława, Tadeusz, et al.
    "LAP: Liability Antibody Profiler by sequence & structural mapping of natural and therapeutic antibodies"

    Examples:
    DeAmdH_NG: Deamidation (with high severity). Motif pattern: Asparagine followed by Glycine.
    FragH_DP: Fragmentation (with high severity). Motif pattern: Aspartate followed by Proline
    """

    DeAmdH_NG = ("N", "G")
    DeAmdH_NS = ("N", "S")
    FragH_DP = ("D", "P")
    Isom_DD = ("D", "D")
    Isom_DG = ("D", "G")
    Isom_DH = ("D", "H")
    Isom_DS = ("D", "S")
    Isom_DT = ("D", "T")
    DeAmdM_NA = ("N", "A")
    DeAmdM_NH = ("N", "H")
    DeAmdM_NN = ("N", "N")
    DeAmdM_NT = ("N", "T")
    Hydro_NP = ("N", "P")
    FragM_TS = ("T", "S")
    TrpOx_W = ("W",)
    MetOx_M = ("M",)
    DeAmdL_SN = ("S", "N")
    DeAmdL_TN = ("T", "N")
    DeAmdL_KN = ("K", "N")
