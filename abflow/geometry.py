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
    Args:
        coords: (..., N_res, 4, 3).
    Returns:
        Dihedral angles in radians from -pi to pi, (..., N_res).
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
