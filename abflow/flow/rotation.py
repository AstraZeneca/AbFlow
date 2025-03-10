"""
Author: Haowen Zhao
Email: hz362@cam.ac.uk

Implementations for SO(3) rotation operations with tensors enforced to be float32.
This script is implemented with heavy help from:
https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/rigid_utils.py
https://github.com/microsoft/protein-frame-flow/blob/main/data/so3_utils.py


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
import torch.nn.functional as F
import numpy as np
from functools import lru_cache, wraps
from einops import rearrange
from typing import Tuple


def enforce_float32(func):
    """
    Decorator to temporarily cast all tensor inputs to float32,
    execute the function, and restore the original dtype for all tensor outputs.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_dtype = next(
            (arg.dtype for arg in args if isinstance(arg, torch.Tensor)), None
        )

        if original_dtype is None:
            raise ValueError("No tensor inputs found to determine the original dtype.")

        args = tuple(
            arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        kwargs = {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        result = func(*args, **kwargs)

        def restore_dtype(output):
            return (
                output.to(original_dtype)
                if isinstance(output, torch.Tensor)
                else output
            )

        if isinstance(result, torch.Tensor):
            return restore_dtype(result)
        elif isinstance(result, (list, tuple)):
            return type(result)(restore_dtype(r) for r in result)
        else:
            return result

    return wrapper


@enforce_float32
def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix"""

    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    
    # Gram-Schmidt orthogonalization
    e1 = F.normalize(a1, dim=-1)
    e2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)
    
    return torch.stack([e1, e2, e3], dim=-1)

@enforce_float32
def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation"""
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)

@enforce_float32
def rot6d_mul(rot6d_1: torch.Tensor, rot6d_2: torch.Tensor) -> torch.Tensor:
    """Compose two 6D rotations"""
    rotmat_1 = rot6d_to_rotmat(rot6d_1)
    rotmat_2 = rot6d_to_rotmat(rot6d_2)
    return rotmat_to_rot6d(torch.einsum('...ij,...jk->...ik', rotmat_1, rotmat_2))

@enforce_float32
def rot6d_inv(rot6d: torch.Tensor) -> torch.Tensor:
    """Invert 6D rotation"""
    rotmat = rot6d_to_rotmat(rot6d)
    return rotmat_to_rot6d(rotmat.transpose(-1, -2))



@enforce_float32
def rot6d_mul_vec(rot6d: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector using 6D representation"""
    rotmat = rot6d_to_rotmat(rot6d)
    return torch.einsum('...ij,...j->...i', rotmat, vec)



@enforce_float32
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


@enforce_float32
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


@enforce_float32
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



@enforce_float32
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


@enforce_float32
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


@enforce_float32
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


@enforce_float32
def rotquat_mul_vec(rotquat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector using a quaternion by performing q * v * q^-1 as in:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#cite_note-8

    :param rotquat: Rotation quaternion tensor of shape [..., 4].
    :param vec: Vector tensor of shape [..., 3].
    :return: Rotated vector tensor of shape [..., 3].
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


@enforce_float32
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


@enforce_float32
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

    :param rotvec: Rotation vector of shape [..., 3].
    :return: Skew-symmetric matrix of shape [..., 3, 3].
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
    :return: Tuple of computed angles, sines of the angles and cosines of angles.
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
    :return: Batch of corresponding rotation matrices.
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


@enforce_float32
def rotvec_to_rotmat(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable rotation vector to matrix conversion"""
    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    small_theta_mask = theta.squeeze(-1) < eps
    
    # Safe axis computation
    axis = torch.where(
        small_theta_mask[..., None],
        F.normalize(rotvec + eps, dim=-1),
        rotvec / (theta + eps)
    )
    
    # Skew-symmetric matrix
    skew = torch.zeros(*rotvec.shape[:-1], 3, 3, device=rotvec.device)
    skew[..., 0, 1] = -axis[..., 2]
    skew[..., 0, 2] = axis[..., 1]
    skew[..., 1, 0] = axis[..., 2]
    skew[..., 1, 2] = -axis[..., 0]
    skew[..., 2, 0] = -axis[..., 1]
    skew[..., 2, 1] = axis[..., 0]

    # Taylor expansion for small angles
    sin_theta = torch.where(
        small_theta_mask[..., None],
        theta - (theta**3)/6,
        torch.sin(theta)
    )
    cos_theta = torch.where(
        small_theta_mask[..., None],
        1 - (theta**2)/2 + (theta**4)/24,
        torch.cos(theta)
    )
    
    # Rodrigues' formula with stabilization
    eye = torch.eye(3, device=rotvec.device).expand_as(skew)
    term1 = eye * cos_theta[..., None]
    term2 = skew * sin_theta[..., None]
    term3 = (axis[..., None] @ axis[..., None, :]) * (1 - cos_theta[..., None])
    
    return term1 + term2 + term3

@enforce_float32
def rotmat_to_rotvec(rotmat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable rotation matrix to vector conversion"""
    trace = torch.einsum('...ii', rotmat)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1+eps, 1-eps)  # Critical clamp
    theta = torch.acos(cos_theta)
    
    skew = 0.5 * (rotmat - rotmat.transpose(-1, -2))
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True) + eps
    
    # Handle θ ≈ 0
    small_theta = theta < eps
    axis = torch.where(
        small_theta[..., None],
        F.normalize(rotmat[..., :, 0], dim=-1),  # Use first column
        axis / axis_norm
    )
    
    # Handle θ ≈ π (antipodal points)
    near_pi = theta > (torch.pi - eps)
    if near_pi.any():
        d_diag = torch.diagonal(rotmat, dim1=-2, dim2=-1)
        axis_pi = torch.sqrt((d_diag - cos_theta[..., None]) / (1 - cos_theta[..., None] + eps))
        axis_pi = torch.where(rotmat[..., 2, 1] < 0, -axis_pi, axis_pi)
        axis = torch.where(near_pi[..., None], axis_pi, axis)
    
    return F.normalize(axis, dim=-1) * theta[..., None]



@enforce_float32
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


@enforce_float32
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



@enforce_float32
def rotquat_to_rotvec(rotquat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation vector (axis-angle representation).

    Formula:
        θ = 2 * arccos(r)
        v = (2 * θ) * (q_v / |q_v|)

    :param rotquat: Quaternion of shape [..., 4].
    :return: Rotation vector of shape [..., 3].
    """

    r = rotquat[..., 0:1]
    v = rotquat[..., 1:]

    theta = 2 * torch.atan2(torch.norm(v, dim=-1)[..., None], r)
    axis = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-7)
    rotvec = axis * theta
    
    return rotvec


@enforce_float32
def rotvec_to_rotquat(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation vector (axis-angle) to a quaternion.

    Formula:
        θ = |v|
        q = [cos(θ / 2), sin(θ / 2) * v / θ]

    :param rotvec: Rotation vector of shape [..., 3].
    :return: Quaternion of shape [..., 4].
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
