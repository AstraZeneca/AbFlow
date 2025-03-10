import torch
import torch.nn.functional as F
import enum

class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT=14;

def create_rotation_matrix(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix from two vectors via the Gram-Schmidt process.

    Vectors in `v1` are taken as the first component of the orthogonal basis, then the component of `v2`
    orthogonal to `v1`, and finally the cross product of `v1` and the orthogonalised `v2`.

    If `v1` or `v2` is invalid (e.g., zero vector or NaN), the function returns the identity matrix.

    :param v1: Tensor of shape (..., 3)
    :param v2: Tensor of shape (..., 3)
    :return: Tensor of shape (..., 3, 3)
    """
    # Ensure inputs are not NaN or infinite
    v1 = torch.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)
    v2 = torch.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute norms of v1 and v2
    v1_norm = torch.linalg.norm(v1, dim=-1, keepdim=True)
    v2_norm = torch.linalg.norm(v2, dim=-1, keepdim=True)

    # Check for invalid vectors (zero norm)
    invalid_v1 = v1_norm.squeeze(-1) < 1e-8  # True if v1 is invalid
    invalid_v2 = v2_norm.squeeze(-1) < 1e-8  # True if v2 is invalid

    # Normalize v1 (handle invalid vectors)
    e1 = torch.where(invalid_v1.unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0], device=v1.device), v1 / torch.clamp(v1_norm, min=1e-8))

    # Orthogonalize v2 with respect to v1 (handle invalid vectors)
    dot_product = torch.sum(e1 * v2, dim=-1, keepdim=True)
    u2 = torch.where(invalid_v1.unsqueeze(-1), v2, v2 - e1 * dot_product)

    # Normalize u2 (handle invalid vectors)
    u2_norm = torch.linalg.norm(u2, dim=-1, keepdim=True)
    e2 = torch.where(invalid_v2.unsqueeze(-1), torch.tensor([0.0, 1.0, 0.0], device=v1.device), u2 / torch.clamp(u2_norm, min=1e-8))

    # Compute the third basis vector (handle invalid vectors)
    e3 = torch.where((invalid_v1 | invalid_v2).unsqueeze(-1), torch.tensor([0.0, 0.0, 1.0], device=v1.device), torch.cross(e1, e2, dim=-1))

    # Stack the basis vectors into a rotation matrix
    rotations = torch.stack([e1, e2, e3], dim=-2).transpose(-2, -1)

    # Replace invalid rotations with the identity matrix
    invalid_rotations = invalid_v1 | invalid_v2
    identity = torch.eye(3, device=v1.device).expand_as(rotations)
    rotations = torch.where(invalid_rotations.unsqueeze(-1).unsqueeze(-1), identity, rotations)

    return rotations


def get_dihedrals(coords: torch.Tensor) -> torch.Tensor:
    """
    :param coords: Tensor of shape (..., N_res, 4, 3)
    :return: Dihedral angles in radians from -pi to pi, (..., N_res)
    """

    # Get the vectors between the atoms
    v0 = coords[..., 1, :] - coords[..., 0, :]
    v1 = coords[..., 2, :] - coords[..., 1, :]
    v2 = coords[..., 3, :] - coords[..., 2, :]

    # Get the normal vectors
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v1, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)

    # Get the sign of the dihedral angle (needs to be computed using normal vectors)
    dihedral_sign = torch.sign((torch.cross(n1, n2, dim=-1) * v1).sum(-1))

    # Get the dihedral angle
    dihedral_angles = dihedral_sign * torch.acos(
                                                (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999)
                                                )

    return dihedral_angles


def compose_rotation_and_translation(
    R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    :param R1: Frame basis, (N, L, 3, 3)
    :param t1: Frame coordinate, (N, L, 3)
    :param R2: Rotation to be applied, (N, L, 3, 3)
    :param t2: Translation to be applied, (N, L, 3)
    :return: R_new <- R1R2, t_new <- R1t2 + t1
    """
    R_new = torch.matmul(R1, R2)  # (N, L, 3, 3)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


def create_chi_rotation(
    dihedrals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converting dihedral angles to rotation matrices around the x-axis.
    See Alphafold2 Supplementary Algorithm 25.

    :param dihedrals: Dihedrals of shape (N_batch, N_res, 5)
    """
    angle_sin, angle_cos = torch.sin(dihedrals), torch.cos(dihedrals)
    angle_sin = angle_sin[..., None, None]  # (N_batch, N_res, 5, 1, 1)
    angle_cos = angle_cos[..., None, None]  # (N_batch, N_res, 5, 1, 1)
    zero = torch.zeros_like(angle_sin)
    one = torch.ones_like(angle_sin)

    row1 = torch.cat([one, zero, zero], dim=-1)  # (N_batch, N_res, 5, 1, 3)
    row2 = torch.cat([zero, angle_cos, -angle_sin], dim=-1)  # (N_batch, N_res, 5, 1, 3)
    row3 = torch.cat([zero, angle_sin, angle_cos], dim=-1)  # (N_batch, N_res, 5, 1, 3)
    chi_rotations = torch.cat([row1, row2, row3], dim=-2)  # (N_batch, N_res, 5, 3, 3)

    chi1_rotations, chi2_rotations, chi3_rotations, chi4_rotations, psi_rotations = (
        chi_rotations.unbind(dim=-3)
    )  # (N_batch, N_res, 3, 3)
    return chi1_rotations, chi2_rotations, chi3_rotations, chi4_rotations, psi_rotations

def construct_3d_basis(center, p1, p2):
    """
    Construct an orthogonal 3D basis given three points.

    Parameters
    ----------
    center : torch.Tensor
        The center point (C_alpha), shape (N, L, 3).
    p1 : torch.Tensor
        First point (C), shape (N, L, 3).
    p2 : torch.Tensor
        Second point (N), shape (N, L, 3).

    Returns
    -------
    torch.Tensor
        Orthogonal 3D basis matrix, shape (N, L, 3, 3).
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)


def project_v2v(v, e, dim):
    """
    Project vector `v` onto vector `e`.

    Parameters
    ----------
    v : torch.Tensor
        Input vector, shape (N, L, 3).
    e : torch.Tensor
        Vector onto which `v` will be projected, shape (N, L, 3).
    dim : int
        Dimension along which to compute the projection.

    Returns
    -------
    torch.Tensor
        Projected vector, shape (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def normalize_vector(v, dim, eps=1e-6):
    """
    Normalize a vector along the specified dimension.

    Parameters
    ----------
    v : torch.Tensor
        Input vector to be normalized.
    dim : int
        Dimension along which to normalize.
    eps : float, optional
        Small epsilon value to avoid division by zero (default is 1e-6).

    Returns
    -------
    torch.Tensor
        Normalized vector.
    """
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)



def global2local(R, t, q):
    """
    Convert global coordinates to local coordinates.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (N, L, 3, 3).
    t : torch.Tensor
        Translation vector, shape (N, L, 3).
    q : torch.Tensor
        Global coordinates, shape (N, L, ..., 3).

    Returns
    -------
    torch.Tensor
        Local coordinates, shape (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.view(N, L, -1, 3).transpose(-1, -2)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))
    p = p.transpose(-1, -2).reshape(q_size)
    return p



def get_3d_basis(center, p1, p2):
    """
    Compute a 3D orthogonal basis given three points.

    Parameters
    ----------
    center : torch.Tensor
        Central point, usually the position of C_alpha, shape (N, L, 3).
    p1 : torch.Tensor
        First point, usually the position of C, shape (N, L, 3).
    p2 : torch.Tensor
        Second point, usually the position of N, shape (N, L, 3).

    Returns
    -------
    torch.Tensor
        Orthogonal basis matrix, shape (N, L, 3, 3).
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)



def get_bb_dihedral_angles(pos_atoms, chain_nb, res_nb, mask_residue):
    """
    Compute backbone dihedral angles (Omega, Phi, Psi) from atomic positions.

    Parameters
    ----------
    pos_atoms : torch.Tensor
        Atomic positions, shape (N, L, A, 3).
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask_residue : torch.Tensor
        Mask for valid residues, shape (N, L).

    Returns
    -------
    tuple of torch.Tensor
        Backbone dihedral angles and their masks, both of shape (N, L, 3).
    """
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_mask, C_term_mask = get_terminus_flag(chain_nb, res_nb, mask_residue)

    omega_mask = torch.logical_not(N_term_mask)
    phi_mask = torch.logical_not(N_term_mask)
    psi_mask = torch.logical_not(C_term_mask)

    omega = F.pad(dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), pad=(1, 0), value=0)
    phi = F.pad(dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]), pad=(1, 0), value=0)
    psi = F.pad(dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]), pad=(0, 1), value=0)

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihed = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed

    return bb_dihed, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    """
    Compute inter-residue Phi and Psi angles.

    Parameters
    ----------
    pos_atoms : torch.Tensor
        Atomic positions, shape (N, L, A, 3).

    Returns
    -------
    torch.Tensor
        Inter-residue Phi and Psi angles, shape (N, L, L, 2).
    """
    N, L = pos_atoms.size()[:2]
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:, :, None, :].expand(N, L, L, 3),
        pos_N[:, None, :, :].expand(N, L, L, 3),
        pos_CA[:, None, :, :].expand(N, L, L, 3),
        pos_C[:, None, :, :].expand(N, L, L, 3)
    )

    ir_psi = dihedral_from_four_points(
        pos_N[:, :, None, :].expand(N, L, L, 3),
        pos_CA[:, :, None, :].expand(N, L, L, 3),
        pos_C[:, :, None, :].expand(N, L, L, 3),
        pos_N[:, None, :, :].expand(N, L, L, 3)
    )

    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)

    return ir_dihed


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Compute flag indicating whether consecutive residues are connected.

    Parameters
    ----------
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask : torch.Tensor
        Mask indicating valid residues, shape (N, L).

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating connected residues, shape (N, L-1).
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])

    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    """
    Identify N-terminus and C-terminus flags for residues.

    Parameters
    ----------
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask : torch.Tensor
        Mask indicating valid residues, shape (N, L).

    Returns
    -------
    tuple of torch.Tensor
        N-terminus and C-terminus flags, both of shape (N, L).
    """
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Compute dihedral angle given four points.

    Parameters
    ----------
    p0, p1, p2, p3 : torch.Tensor
        Coordinates of four points, shape (*, 3).

    Returns
    -------
    torch.Tensor
        Dihedral angles in radians, shape (*,).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2

    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)

    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)

    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed