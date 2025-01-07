"""
Author: Haowen Zhao
Email: hz362@cam.ac.uk

Implementations for SO(3) rotation operations.

### Rotation Representations:
1. **Rotation Matrix**: 3x3 orthogonal matrix with determinant 1.
2. **Quaternion (Cayley-Klein Parameters)**: 4D vector [r, i, j, k] with unit length (r^2 + i^2 + j^2 + k^2 = 1).
3. **Rotation Vector (Axis-Angle)**: 3D vector [x, y, z] with unit direction for 
the rotation axis and magnitude for the rotation angle.

### Usage Examples:

- **Inverting a Rotation:**
```python
rotmat_inv = rotmat_inv(rotmat)
rotquat_inv = rotquat_inv(rotquat)
rotvec_inv = rotvec_inv(rotvec)
```

- **Multiplying Rotations (left or right multiplication):**
```python
rotmat_result = rotmats_mul(rotmat1, rotmat2)
rotvec_result = rotvecs_mul(rotvec1, rotvec2)
rotquat_result = rotquats_mul(rotquat1, rotquat2)
```

- **Rotating a Vector (left multiplication):**
```python
rotated_vec = rotmat_mul_vec(rotmat, vec)
rotated_vec = rotvec_mul_vec(rotvec, vec)
rotated_vec = rotquat_mul_vec(rotquat, vec)
```

- **Converting Between Rotation Representations:**
```python
rotquat = rotmat_to_rotquat(rotmat)
rotvec = rotmat_to_rotvec(rotmat)
rotmat = rotvec_to_rotmat(rotvec)
rotmat = rotquat_to_rotmat(rotquat)
rotvec = rotquat_to_rotvec(rotquat)
rotquat = rotvec_to_rotquat(rotvec)
```
"""

import torch
import numpy as np
from functools import lru_cache
from einops import rearrange
from typing import Tuple


def rotmat_inv(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Invert a rotation matrix by taking its transpose.

    Formula:
        R^(-1) = R^T s.t. R * R^T = I

    :param rotmat: Input rotation matrix of shape [..., 3, 3].
    :return: Inverted rotation matrix of shape [..., 3, 3].
    """
    rotmat_inv = rearrange(rotmat, "... i j -> ... j i")
    return rotmat_inv


def rotquat_inv(rotquat: torch.Tensor) -> torch.Tensor:
    """
    Invert a unit quaternion in [r, i, j, k] format by taking its conjugate.

    Formula:
        q^(-1) = [r, -i, -j, -k]

    :param rotquat: Input quaternion of shape [..., 4].
    :return: Inverted quaternion of shape [..., 4].
    """
    rotquat_inv = rotquat.clone()
    rotquat_inv[..., 1:] = -rotquat_inv[..., 1:]
    return rotquat_inv


def rotvec_inv(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Invert a rotation vector by negating it (rotating about the opposite
    direction of the original axis, with the same angle).

    Formula:
        r^(-1) = -r

    :param rotvec: Input rotation vector of shape [..., 3].
    :return: Inverted rotation vector of shape [..., 3].
    """
    rotvec_inv = -rotvec
    return rotvec_inv


def rotmats_mul(rotmat1: torch.Tensor, rotmat2: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written
    out by hand to avoid AMP downcasting.

    Formula:
        R = R1 * R2
        [R]_ij = sum_k R1_ik * R2_kj

    :param rotmat1: First rotation matrix tensor of shape [..., 3, 3].
    :param rotmat2: Second rotation matrix tensor of shape [..., 3, 3].
    :return: Resultant rotation matrix tensor of shape [..., 3, 3].
    """

    def row_mul(i):
        return torch.stack(
            [
                rotmat1[..., i, 0] * rotmat2[..., 0, 0]
                + rotmat1[..., i, 1] * rotmat2[..., 1, 0]
                + rotmat1[..., i, 2] * rotmat2[..., 2, 0],
                rotmat1[..., i, 0] * rotmat2[..., 0, 1]
                + rotmat1[..., i, 1] * rotmat2[..., 1, 1]
                + rotmat1[..., i, 2] * rotmat2[..., 2, 1],
                rotmat1[..., i, 0] * rotmat2[..., 0, 2]
                + rotmat1[..., i, 1] * rotmat2[..., 1, 2]
                + rotmat1[..., i, 2] * rotmat2[..., 2, 2],
            ],
            dim=-1,
        )

    rotmats = torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )

    return rotmats


def rotvecs_mul(rotvec1: torch.Tensor, rotvec2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two rotations represented as rotation vectors (axis-angle) by converting
    them to rotation matrices, multiplying the matrices, and converting back to a
    rotation vector.

    Rotation vectors represent rotations as an axis and an angle, but the composition
    (multiplication) of two rotations cannot be done directly in this representation.

    :param rotvec1: First rotation vector tensor of shape [..., 3].
    :param rotvec2: Second rotation vector tensor of shape [..., 3].
    :return: Resultant rotation vector tensor of shape [..., 3].
    """

    rotmat1 = rotvec_to_rotmat(rotvec1)
    rotmat2 = rotvec_to_rotmat(rotvec2)

    rotmats = rotmats_mul(rotmat1, rotmat2)
    rotvecs = rotmat_to_rotvec(rotmats)

    return rotvecs


def rotquats_mul(rotquat1: torch.Tensor, rotquat2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Definition:
        q1 = r1 + i1 * i + j1 * j + k1 * k = r1 + v1
        q2 = r2 + i2 * i + j2 * j + k2 * k = r2 + v2
        q = q1 * q2 = r1 * r2 + r1 * v2 + r2 * v1 + v1 * v2
    Quaternion rules:
        i^2 = j^2 = k^2 = ijk = -1, ij = -ji = k, jk = -kj = i, ki = -ik = j
    Expanding v1 * v2 using quaternion rules:
        real part: - dot(v1, v2)
        vector part: (j1 * k2 - k1 * j2) i + (k1 * i2 - i1 * k2) j + (i1 * j2 - j1 * i2) k
                        = cross(v1, v2)
    Formula:
        q = q1 * q2 = [r1 * r2 - dot(v1, v2), r1 * v2 + r2 * v1 + cross(v1, v2)]

    :param rotquat1: First quaternion tensor of shape [..., 4].
    :param rotquat2: Second quaternion tensor of shape [..., 4].
    :return: Resultant quaternion tensor of shape [..., 4].
    """
    r1, v1 = rotquat1[..., 0:1], rotquat1[..., 1:]
    r2, v2 = rotquat2[..., 0:1], rotquat2[..., 1:]

    scalar_part = r1 * r2 - torch.sum(v1 * v2, dim=-1, keepdim=True)
    vector_part = r1 * v2 + r2 * v1 + torch.cross(v1, v2, dim=-1)

    rotquats = torch.cat([scalar_part, vector_part], dim=-1)
    return rotquats


def rotmat_mul_vec(rotmat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Formula:
        v' = R * v
        v'_i = sum_j R_ij * v_j

    :param rotmat: Rotation matrix tensor of shape [..., 3, 3].
    :param vec: Vector tensor of shape [..., 3].
    :return: Rotated vector tensor of shape [..., 3].
    """
    x, y, z = torch.unbind(vec, dim=-1)
    rotated_vec = torch.stack(
        [
            rotmat[..., 0, 0] * x + rotmat[..., 0, 1] * y + rotmat[..., 0, 2] * z,
            rotmat[..., 1, 0] * x + rotmat[..., 1, 1] * y + rotmat[..., 1, 2] * z,
            rotmat[..., 2, 0] * x + rotmat[..., 2, 1] * y + rotmat[..., 2, 2] * z,
        ],
        dim=-1,
    )
    return rotated_vec


def rotvec_mul_vec(
    rotvec: torch.Tensor, vec: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """
    Rotate a vector using a rotation vector (axis-angle representation) via
    Rodrigues' rotation formula as in:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Given a rotation vector `r`, where the direction represents the axis of
    rotation k and the magnitude represents the rotation angle θ, the rotated
    vector `v_rot` for a given input vector `v` is computed as:

        v_rot = v * cos(θ) + (k x v) * sin(θ) + k * (k ⋅ v) * (1 - cos(θ))

    Where:
        - `k` is the normalized axis of rotation (`r / |r|`)
        - `θ` is the magnitude of the rotation vector (`|r|`)
        - `x` denotes the cross product
        - `⋅` denotes the dot product

    For small angles < tol, the rotation vector is approximated as the identity.

    :param rotvec: Rotation vector of shape [..., 3].
    :param vec: Input vector to be rotated of shape [..., 3].
    :return: Rotated vector of shape [..., 3].
    """

    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    small_angle_mask = theta < tol
    axis = rotvec / (theta + tol)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    term1 = vec * cos_theta
    term2 = torch.cross(axis, vec) * sin_theta
    term3 = axis * (torch.sum(axis * vec, dim=-1, keepdim=True)) * (1 - cos_theta)

    rotated_vec = term1 + term2 + term3
    rotated_vec = torch.where(small_angle_mask, vec, rotated_vec)

    return rotated_vec


def rotquat_mul_vec(rotquat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector using a quaternion by performing q * v * q^-1 as in:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#cite_note-8

    :param rotquat: Rotation quaternion of shape [..., 4].
    :param vec: Vector to be rotated of shape [..., 3].
    :return: Rotated vector of shape [..., 3].
    """

    vec_quat = torch.cat([torch.zeros_like(vec[..., :1]), vec], dim=-1)
    quat_conjugate = torch.cat([rotquat[..., :1], -rotquat[..., 1:]], dim=-1)

    intermediate_quat = rotquats_mul(rotquat, vec_quat)
    rotated_vec_quat = rotquats_mul(intermediate_quat, quat_conjugate)

    rotated_vec = rotated_vec_quat[..., 1:]

    return rotated_vec


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])

_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

_CACHED_QUATS = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}


@lru_cache(maxsize=None)
def _get_quat(quat_key, dtype, device):
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)


def rotmat_to_rotquat(rotmat: torch.Tensor) -> torch.Tensor:

    rotmat = [[rotmat[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rotmat

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    rotquat = vectors[..., -1]

    return rotquat


def rotquat_to_rotmat(rotquat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    :param rotquat: [..., 4] quaternions
    :return: [..., 3, 3] rotation matrices
    """
    # [*, 4, 4]
    rotquat = rotquat[..., None] * rotquat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=rotquat.dtype, device=rotquat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(rotquat.shape[:-2]) + mat.shape)
    rotquat = rotquat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(rotquat, dim=(-3, -4))


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3 by 3 identity matrix and broadcast it to a batch of target matrices.

    :param target: Batch of target 3 by 3 matrices.
    :return: 3 by 3 identity matrices in the shapes of the target.
    """
    id3 = torch.eye(3, device=target.device, dtype=target.dtype)
    id3 = torch.broadcast_to(id3, target.shape)
    return id3


def rotvec_to_skewmat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation vector to a skew-symmetric so(3) matrix.

    Formula:
                    [  0 -z  y]
        [x,y,z] ->  [  z  0 -x]
                    [ -y  x  0]

    :param rotvec: Rotation vector tensor of shape [..., 3].
    :return: Skew-symmetric matrix tensor of shape [..., 3, 3].
    """

    # Generate empty skew matrices.
    skew_matrices = torch.zeros(
        (*rotvec.shape, 3), device=rotvec.device, dtype=rotvec.dtype
    )

    # Populate positive values.
    skew_matrices[..., 2, 1] = rotvec[..., 0]
    skew_matrices[..., 0, 2] = rotvec[..., 1]
    skew_matrices[..., 1, 0] = rotvec[..., 2]

    # Generate skew symmetry.
    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)

    return skew_matrices


def skewmat_to_rotvec(skew_mat: torch.Tensor) -> torch.Tensor:
    """
    Convert a skew-symmetric matrix to a rotation vector.

    Formula:
        [  0 -z  y]
        [  z  0 -x] -> [x, y, z]
        [ -y  x  0]
    """

    vectors = torch.zeros_like(skew_mat[..., 0])
    vectors[..., 0] = skew_mat[..., 2, 1]
    vectors[..., 1] = skew_mat[..., 0, 2]
    vectors[..., 2] = skew_mat[..., 1, 0]
    return vectors


def angle_from_rotmat(
    rotation_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rotation angles (as well as their sines and cosines) encoded by rotation matrices.
    Uses atan2 for better numerical stability for small angles.

    :param rotation_matrices: Batch of rotation matrices.
    :return: Tuple of rotation angles, sines of angles, and cosines of angles.
    """
    # Compute sine of angles (uses the relation that the unnormalized skew vector generated by a
    # rotation matrix has the length 2*sin(theta))
    skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
    skew_vectors = skewmat_to_rotvec(skew_matrices)
    angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
    # Compute the cosine of the angle using the relation cos theta = 1/2 * (Tr[R] - 1)
    angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0

    # Compute angles using the more stable atan2
    angles = torch.atan2(angles_sin, angles_cos)

    return angles, angles_sin, angles_cos


def skewmat_exponential_map(
    angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation vector in skew matrix representation. Maps the
    rotation from the lie group to the rotation matrix representation. Uses the following form of
    Rodrigues' formula instead of `torch.linalg.matrix_exp` for better computational performance
    (in this case the skew matrix already contains the angle factor):

    Formula:
        exp(K) = I + sin(theta) / theta * K + (1 - cos(theta)) / theta^2 * K^2

    This form has the advantage, that Taylor expansions can be used for small angles (instead of
    having to compute the unit length axis by dividing the rotation vector by small angles):

        sin(theta) / theta = 1 - theta^2 / 6
        (1 - cos(theta)) / theta^2 = 1 / 2 - theta^2 / 24

    :param angles: Batch of rotation angles.
    :param skew_matrices: Batch of rotation axes in skew matrix (lie so(3)) basis.
    :param tol: small offset for numerical stability.
    :return: Batch of rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angles and pre-compute square.
    angles = angles[..., None, None]
    angles_sq = angles.square()

    # Get standard terms.
    sin_coeff = torch.sin(angles) / angles
    cos_coeff = (1.0 - torch.cos(angles)) / angles_sq
    # Use second order Taylor expansion for values close to zero.
    sin_coeff_small = 1.0 - angles_sq / 6.0
    cos_coeff_small = 0.5 - angles_sq / 24.0

    mask_zero = torch.abs(angles) < tol
    sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
    cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)

    # Compute matrix exponential using Rodrigues' formula.
    exp_skew = (
        id3
        + sin_coeff * skew_matrices
        + cos_coeff * torch.einsum("...ik,...kj->...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def rotmat_to_rotvec(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to rotation vectors (logarithmic map from SO(3) to so(3)).
    The standard logarithmic map can be derived from Rodrigues' formula via Taylor approximation
    (in this case operating on the vector coefficients of the skew so(3) basis).

    Formula:


    ..math ::

        \left[\log(\mathbf{R})\right]^\lor = \frac{\theta}{2\sin(\theta)} \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    This formula has problems at 1) angles theta close or equal to zero and 2) at angles close and
    equal to pi.

    To improve numerical stability for case 1), the angle term at small or zero angles is
    approximated by its truncated Taylor expansion:

    .. math ::

        \left[\log(\mathbf{R})\right]^\lor \approx \frac{1}{2} (1 + \frac{\theta^2}{6}) \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    For angles close or equal to pi (case 2), the outer product relation can be used to obtain the
    squared rotation vector:

    .. math :: \omega \otimes \omega = \frac{1}{2}(\mathbf{I} + R)

    Taking the root of the diagonal elements recovers the normalized rotation vector up to the signs
    of the component. The latter can be obtained from the off-diagonal elements.

    Adapted from https://github.com/jasonkyuyim/se3_diffusion/blob/2cba9e09fdc58112126a0441493b42022c62bbea/data/so3_utils.py
    which was adapted from https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py
    with heavy help from https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

    :param rotmat: Batch of rotation matrices.
    :return: Batch of rotation vectors.
    """
    # Get angles and sin/cos from rotation matrix.
    angles, angles_sin, _ = angle_from_rotmat(rotmat)
    # Compute skew matrix representation and extract so(3) vector components.
    vector = skewmat_to_rotvec(rotmat - rotmat.transpose(-2, -1))

    # Three main cases for angle theta, which are captured
    # 1) Angle is 0 or close to zero -> use Taylor series for small values / return 0 vector.
    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    # 2) Angle is close to pi -> use outer product relation.
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(
        angles.dtype
    )
    # 3) Angle is unproblematic -> use the standard formula.
    mask_else = (1 - mask_zero) * (1 - mask_pi)

    # Compute case dependent pre-factor (1/2 for angle close to 0, angle otherwise).
    numerator = mask_zero / 2.0 + angles * mask_else
    # The Taylor expansion used here is actually the inverse of the Taylor expansion of the inverted
    # fraction sin(x) / x which gives better accuracy over a wider range (hence the minus and
    # position in denominator).
    denominator = (
        (1.0 - angles**2 / 6.0) * mask_zero  # Taylor expansion for small angles.
        + 2.0 * angles_sin * mask_else  # Standard formula.
        + mask_pi  # Avoid zero division at angle == pi.
    )
    prefactor = numerator / denominator
    vector = vector * prefactor[..., None]

    # For angles close to pi, derive vectors from their outer product (ww' = 1 + R).
    id3 = _broadcast_identity(rotmat)
    skew_outer = (id3 + rotmat) / 2.0
    # Ensure diagonal is >= 0 for square root (uses identity for masking).
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3

    # Get basic rotation vector as sqrt of diagonal (is unit vector).
    vector_pi = torch.sqrt(torch.diagonal(skew_outer, dim1=-2, dim2=-1))

    # Compute the signs of vector elements (up to a global phase).
    # Fist select indices for outer product slices with the largest norm.
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    # Select rows of outer product and determine signs.
    signs_line = torch.take_along_dim(
        skew_outer, dim=-2, indices=signs_line_idx[..., None, None]
    )
    signs_line = signs_line.squeeze(-2)
    signs = torch.sign(signs_line)

    # Apply signs and rotation vector.
    vector_pi = vector_pi * angles[..., None] * signs

    # Fill entries for angle == pi in rotation vector (basic vector has zero entries at this point).
    rotvec = vector + vector_pi * mask_pi[..., None]

    return rotvec


def rotvec_to_rotmat(rotvec: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """
    Convert rotation vectors to rotation matrix representation. The length of the rotation vector
    is the angle of rotation, the unit vector the rotation axis.

    :param rotvec: Batch of rotation vectors.
    :param tol: small offset for numerical stability.
    :return: Batch of rotation matrices.
    """
    # Compute rotation angle as vector norm.
    rotation_angles = torch.norm(rotvec, dim=-1)

    # Map axis to skew matrix basis.
    skew_matrices = rotvec_to_skewmat(rotvec)

    # Compute rotation matrices via matrix exponential.
    rotmat = skewmat_exponential_map(rotation_angles, skew_matrices, tol=tol)

    return rotmat


def rotquat_to_rotvec(rotquat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation vector (axis-angle representation).

    Formula:
        θ = 2 * arccos(r)
        v = (2 * θ) * (q_v / |q_v|)

    :param rotquat: Quaternion of shape [..., 4] in [r, i, j, k] format.
    :return: Rotation vector of shape [..., 3].
    """
    r = rotquat[..., 0:1]
    v = rotquat[..., 1:]

    theta = 2 * torch.atan2(torch.norm(v, dim=-1), r)
    axis = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-7)
    rotvec = axis * theta[..., None]

    return rotvec


def rotvec_to_rotquat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation vector (axis-angle) to a quaternion.

    Formula:
        θ = |v|
        q = [cos(θ / 2), sin(θ / 2) * v / θ]

    :param rotvec: Rotation vector of shape [..., 3].
    :return: Quaternion of shape [..., 4] in [r, i, j, k] format.
    """

    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    small_angle_mask = theta < 1e-7
    axis = rotvec / (theta + 1e-7)

    r = torch.cos(theta / 2)
    v = axis * torch.sin(theta / 2)

    r = torch.where(small_angle_mask, torch.ones_like(r), r)
    v = torch.where(small_angle_mask, torch.zeros_like(v), v)

    rotquat = torch.cat([r, v], dim=-1)

    return rotquat
