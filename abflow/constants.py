"""
Contains constants used in protein structure analysis.
"""

from enum import Enum, IntEnum
from Bio.PDB.Polypeptide import protein_letters_3to1

# Conversion scales between nanometers and angstroms
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

# Residue types
RES_TYPE_NUMBER = 22
MASK_TOKEN = 20
PAD_TOKEN = 21

# Chain types
CHAIN_TYPE_NUMBER = 5


# stores indices under the 3-letter code of each AA
class AminoAcid3(Enum):
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


# stores indices under the 1-letter code of each AA
AminoAcid1 = Enum(
    "AminoAcid1",
    {
        protein_letters_3to1[aa]: index.value
        for aa, index in AminoAcid3.__members__.items()
    },
)


# Mappings between amino acid indices and their 3-letter and 1-letter codes
aa3_index_to_name = {v.value: k for k, v in AminoAcid3.__members__.items()}
aa1_index_to_name = {v.value: k for k, v in AminoAcid1.__members__.items()}
aa3_name_to_index = {k: v.value for k, v in AminoAcid3.__members__.items()}
aa1_name_to_index = {k: v.value for k, v in AminoAcid1.__members__.items()}

chi_angles_atoms = {
    AminoAcid3.ALA.value: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AminoAcid3.ARG.value: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    AminoAcid3.ASN.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AminoAcid3.ASP.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    AminoAcid3.CYS.value: [["N", "CA", "CB", "SG"]],
    AminoAcid3.GLN.value: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    AminoAcid3.GLU.value: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    AminoAcid3.GLY.value: [],
    AminoAcid3.HIS.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    AminoAcid3.ILE.value: [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    AminoAcid3.LEU.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.LYS.value: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    AminoAcid3.MET.value: [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    AminoAcid3.PHE.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.PRO.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    AminoAcid3.SER.value: [["N", "CA", "CB", "OG"]],
    AminoAcid3.THR.value: [["N", "CA", "CB", "OG1"]],
    AminoAcid3.TRP.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.TYR.value: [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    AminoAcid3.VAL.value: [["N", "CA", "CB", "CG1"]],
}


chi_angles_mask = {
    AminoAcid3.ALA.value: [False, False, False, False],  # ALA
    AminoAcid3.ARG.value: [True, True, True, True],  # ARG
    AminoAcid3.ASN.value: [True, True, False, False],  # ASN
    AminoAcid3.ASP.value: [True, True, False, False],  # ASP
    AminoAcid3.CYS.value: [True, False, False, False],  # CYS
    AminoAcid3.GLN.value: [True, True, True, False],  # GLN
    AminoAcid3.GLU.value: [True, True, True, False],  # GLU
    AminoAcid3.GLY.value: [False, False, False, False],  # GLY
    AminoAcid3.HIS.value: [True, True, False, False],  # HIS
    AminoAcid3.ILE.value: [True, True, False, False],  # ILE
    AminoAcid3.LEU.value: [True, True, False, False],  # LEU
    AminoAcid3.LYS.value: [True, True, True, True],  # LYS
    AminoAcid3.MET.value: [True, True, True, False],  # MET
    AminoAcid3.PHE.value: [True, True, False, False],  # PHE
    AminoAcid3.PRO.value: [True, True, False, False],  # PRO
    AminoAcid3.SER.value: [True, False, False, False],  # SER
    AminoAcid3.THR.value: [True, False, False, False],  # THR
    AminoAcid3.TRP.value: [True, True, False, False],  # TRP
    AminoAcid3.TYR.value: [True, True, False, False],  # TYR
    AminoAcid3.VAL.value: [True, False, False, False],  # VAL
}

restype_to_heavyatom_names = {
    AminoAcid3.ALA.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.ARG.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.ASN.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.ASP.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.CYS.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "SG",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.GLN.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.GLU.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.GLY.value: [
        "N",
        "CA",
        "C",
        "O",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.HIS.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.ILE.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.LEU.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.LYS.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.MET.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.PHE.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.PRO.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.SER.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.THR.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.TRP.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ],
    AminoAcid3.TYR.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
        "OXT",
    ],
    AminoAcid3.VAL.value: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "OXT",
    ],
}

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name != ""}
    for resname, atoms in restype_to_heavyatom_names.items()
}


class CDRName(Enum):
    """
    Enumeration for Complementarity-Determining Regions (CDRs) in antibodies.
    """

    NONCDR = 0
    HCDR1 = 1
    HCDR2 = 2
    HCDR3 = 3
    LCDR1 = 4
    LCDR2 = 5
    LCDR3 = 6


# Mappings between CDR indices and their names
cdr_index_to_name = {cdr.value: cdr.name for cdr in CDRName}
cdr_name_to_index = {cdr.name: cdr.value for cdr in CDRName}


class BondLengths(Enum):
    """
    Enumeration for backbone bond lengths in proteins.

    Values are taken from:
    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 1.459
    CA_C = 1.525
    C_N = 1.336
    C_O = 1.229


class BondLengthStdDevs(Enum):
    """
    Enum for storing standard deviations in backbone bond lengths, taken from:

    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA = 0.020
    CA_C = 0.026
    C_N = 0.023


class BondAngles(Enum):
    """
    Enumeration for backbone bond angles in proteins.

    Values are taken from:
    Engh, R.A., & Huber, R. (2006).
    Structure quality and target parameters. International Tables for Crystallography.
    """

    N_CA_C = 1.937
    CA_C_N = 2.046
    C_N_CA = 2.124


class BondAngleStdDevs(Enum):
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


class GlyBBCoords(Enum):
    """
    Enumeration for ideal backbone coordinates of glycine in a protein.
    """

    N = tuple([-0.527, 1.337, 0.0])
    CA = tuple([0.0, 0.0, 0.0])
    C = tuple([1.517, 0.0, 0.0])
    CB = tuple([0.0, 0.0, 0.0])


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
