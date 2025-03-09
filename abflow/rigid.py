# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from functools import lru_cache
from typing import Tuple, Any, Sequence, Callable, Optional

import numpy as np
import torch
from abflow.flow.rotation import rot6d_mul_vec, rot6d_inv

def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written
    out by hand to avoid AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab
    """

    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad
    )
    return trans


@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    quat = torch.zeros(
        (*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad
    )

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


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


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
):
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

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
    return vectors[..., -1]


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


def quat_multiply(quat1, quat2):
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None],
        dim=(-3, -2),
    )


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2)
    )


def invert_rot_mat(rot_mat: torch.Tensor):
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


class Rotation:
    def __init__(self, rot6d: torch.Tensor):
        """
        Initialize with 6D rotation representation.
        Args:
            rot6d: [*, 6] tensor
        """
        self._rot6d = rot6d

    @property
    def rot6d(self) -> torch.Tensor:
        """Get the 6D rotation representation."""
        return self._rot6d

    @property
    def rotmat(self) -> torch.Tensor:
        """Convert 6D to 3x3 rotation matrix."""
        return rot6d_to_rotmat(self._rot6d)

    def compose(self, other: Rotation) -> Rotation:
        """
        Compose two rotations (self followed by other).
        Args:
            other: Another Rotation object
        Returns:
            A new Rotation object representing the composition
        """
        return Rotation(rot6d_mul(self.rot6d, other.rot6d))

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to points.
        Args:
            pts: [*, 3] tensor of points
        Returns:
            Rotated points [*, 3]
        """
        return rot6d_mul_vec(self.rot6d, pts)

    def invert(self) -> Rotation:
        """Invert the rotation."""
        return Rotation(rot6d_inv(self.rot6d))

    def unsqueeze(self, dim: int) -> Rotation:
        """
        Unsqueeze the rotation along a dimension.
        Args:
            dim: Dimension to unsqueeze
        Returns:
            Unsqueezed Rotation object
        """
        return Rotation(self.rot6d.unsqueeze(dim))

    def detach(self) -> Rotation:
        """Detach the rotation from the computation graph."""
        return Rotation(self.rot6d.detach())

    @classmethod
    def identity(cls, shape: Tuple[int], dtype: torch.dtype, device: torch.device):
        """
        Create an identity rotation.
        Args:
            shape: Shape of the batch dimensions
            dtype: Data type
            device: Device
        Returns:
            Identity Rotation object
        """
        id_rot6d = torch.zeros(*shape, 6, dtype=dtype, device=device)
        id_rot6d[..., 0] = 1.0
        id_rot6d[..., 4] = 1.0  # Identity in 6D
        return cls(id_rot6d)

    def map_tensor_fn(self, fn: Callable[torch.Tensor, torch.Tensor]) -> Rotation:
        """
        Apply a function to the underlying 6D tensor.
        Args:
            fn: Tensor -> Tensor function
        Returns:
            Transformed Rotation object
        """
        return Rotation(fn(self.rot6d))

    def cuda(self) -> Rotation:
        """Move rotation to CUDA device."""
        return Rotation(self.rot6d.cuda())

    def to(self, device: torch.device) -> Rotation:
        """Move rotation to specified device."""
        return Rotation(self.rot6d.to(device))

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = rot6d_inv(rot_mats)
        return rot6d_mul_vec(inv_rot_mats, pts)


    def get_rot_mats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        return self._rot6d


class Rigid:
    def __init__(self, rots: Rotation, trans: torch.Tensor):
        """
        Initialize a rigid transformation.
        Args:
            rots: Rotation object
            trans: [*, 3] translation tensor
        """
        self._rots = rots
        self._trans = trans

    @staticmethod
    def from_6d(rot6d: torch.Tensor, trans: torch.Tensor) -> Rigid:
        """
        Create a Rigid object from 6D rotation and translation.
        Args:
            rot6d: [*, 6] rotation tensor
            trans: [*, 3] translation tensor
        Returns:
            Rigid object
        """
        return Rigid(Rotation(rot6d), trans)

    def get_rots(self) -> Rotation:
        """Get the rotation component."""
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """Get the translation component."""
        return self._trans

    def compose(self, other: Rigid) -> Rigid:
        """
        Compose two rigid transformations (self followed by other).
        Args:
            other: Another Rigid object
        Returns:
            Composed Rigid object
        """
        new_rot = self._rots.compose(other._rots)
        new_trans = self._rots.apply(other._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to points.
        Args:
            pts: [*, 3] tensor of points
        Returns:
            Transformed points [*, 3]
        """
        return self._rots.apply(pts) + self._trans

    def invert(self) -> Rigid:
        """Invert the transformation."""
        inv_rot = self._rots.invert()
        inv_trans = -inv_rot.apply(self._trans)
        return Rigid(inv_rot, inv_trans)

    def scale_translation(self, factor: float) -> Rigid:
        """
        Scale the translation component.
        Args:
            factor: Scaling factor
        Returns:
            Scaled Rigid object
        """
        return Rigid(self._rots, self._trans * factor)

    def __getitem__(self, index: Any) -> Rigid:
        """
        Index the transformation.
        Args:
            index: Index or slice
        Returns:
            Indexed Rigid object
        """
        if type(index) != tuple:
            index = (index,)
        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)]
        )

    def __mul__(self, right: torch.Tensor) -> Rigid:
        """
        Pointwise multiplication with a tensor.
        Args:
            right: Tensor multiplicand
        Returns:
            Transformed Rigid object
        """
        if not isinstance(right, torch.Tensor):
            raise TypeError("Multiplicand must be a Tensor")
        return Rigid(
            self._rots.map_tensor_fn(lambda x: x * right),
            self._trans * right[..., None]
        )

    def __rmul__(self, left: torch.Tensor) -> Rigid:
        """Reverse pointwise multiplication."""
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """Get the shape of the transformation."""
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        """Get the device of the transformation."""
        return self._trans.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the transformation."""
        return self._rots.rot6d.dtype

    def cuda(self) -> Rigid:
        """Move transformation to CUDA device."""
        return Rigid(self._rots.cuda(), self._trans.cuda())

    def to(self, device: torch.device) -> Rigid:
        """Move transformation to specified device."""
        return Rigid(self._rots.to(device), self._trans.to(device))

    @staticmethod
    def identity(shape: Tuple[int], dtype: torch.dtype, device: torch.device) -> Rigid:
        """
        Create an identity transformation.
        Args:
            shape: Shape of the batch dimensions
            dtype: Data type
            device: Device
        Returns:
            Identity Rigid object
        """
        return Rigid(
            Rotation.identity(shape, dtype, device),
            torch.zeros(*shape, 3, dtype=dtype, device=device)
        )

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        """
        Create from 4x4 transformation matrix.
        Args:
            t: [*, 4, 4] tensor
        Returns:
            Rigid object
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Input must be 4x4 matrices")
        return Rigid(
            Rotation(rotmat_to_rot6d(t[..., :3, :3])),
            t[..., :3, 3]
        )

    @staticmethod
    def from_tensor_7(t: torch.Tensor, normalize_quats: bool = False) -> Rigid:
        """
        Create from 7D tensor (quaternion + translation).
        Args:
            t: [*, 7] tensor
            normalize_quats: Whether to normalize quaternions
        Returns:
            Rigid object
        """
        if t.shape[-1] != 7:
            raise ValueError("Input must have 7 dimensions")
        quats, trans = t[..., :4], t[..., 4:]
        if normalize_quats:
            quats = quats / torch.norm(quats, dim=-1, keepdim=True)
        return Rigid(
            Rotation(rotquat_to_rot6d(quats)),
            trans
        )

    @staticmethod
    def from_3_points(p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float = 1e-8) -> Rigid:
        """
        Create from 3 points using Gram-Schmidt.
        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon for numerical stability
        Returns:
            Rigid object
        """
        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return Rigid(Rotation(rotmat_to_rot6d(rots)), torch.stack(origin, dim=-1))
    
    
    def unsqueeze(
        self,
        dim: int,
    ) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the
        shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim)

        return Rigid(rots, trans)


    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts)