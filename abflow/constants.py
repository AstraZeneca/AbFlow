"""
Contains constants used in protein structure analysis.
"""

from enum import Enum
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
