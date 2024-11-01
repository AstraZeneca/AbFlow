import torch


def create_rotation_matrix(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix from two vectors via the Gram-Schmidt process.

    Vectors in `v1` are taken as the first component of the orthogonal basis, then the component of `v2`
    orthogonal to `v1`, and finally the cross product of `v1` and the orthogonalised `v2`.

    :param v1: Tensor of shape (..., 3)
    :param v2: Tensor of shape (..., 3)
    :return: Tensor of shape (..., 3, 3)
    """

    e1 = v1 / torch.linalg.norm(v1, dim=-1).unsqueeze(-1)
    u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1).unsqueeze(-1))
    e2 = u2 / torch.linalg.norm(u2, dim=-1).unsqueeze(-1)
    e3 = torch.cross(e1, e2, dim=-1)

    rotations = torch.stack([e1, e2, e3], dim=-2).transpose(-2, -1)
    rotations = torch.nan_to_num(rotations)

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

    # Get the sign of the dihedral angle
    dihedral_sign = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))

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


def create_psi_chi_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Compute psi and chi rotation matrices from torsional angles.

    Here we provide angles instead of alpha in af2 between (0,2pi)

    See alphafold supplementary Algorithm 25 for details.

    :param angles: (B, N, 5), angles between (0,2pi)
    :return: Torsional angle rotation matrices, (B, N, 5, 3, 3).
    """
    batch_size, n_res = angles.shape[:2]
    sine, cosine = torch.sin(angles), torch.cos(angles)
    sine = sine.reshape(batch_size, n_res, -1, 1, 1)
    cosine = cosine.reshape(batch_size, n_res, -1, 1, 1)
    zero = torch.zeros_like(sine)
    one = torch.ones_like(sine)

    row1 = torch.cat([one, zero, zero], dim=-1)  # (B, N, 5, 1, 3)
    row2 = torch.cat([zero, cosine, -sine], dim=-1)  # (B, N, 5, 1, 3)
    row3 = torch.cat([zero, sine, cosine], dim=-1)  # (B, N, 5, 1, 3)
    R = torch.cat([row1, row2, row3], dim=-2)  # (B, N, 5, 3, 3)

    return R
