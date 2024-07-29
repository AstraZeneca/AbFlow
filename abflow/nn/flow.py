"""
Contains the flow matching frameworks for 3D coordinates, probability simplex and SO(3) rotations.
"""

import torch
from typing import Union
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation

from .so3_utils import geodesic_t, calc_rot_vf


class EuclideanFlow(ABC):
    """
    A linear flow matching framework for Euclidean space.
    """

    def __init__(self):

        pass

    @abstractmethod
    def _prior_sampling(
        self,
        num_batch: int,
        num_res: int,
        device: torch.device,
    ) -> torch.Tensor:

        pass

    def init(self, num_batch: int, num_res: int, device: torch.device) -> torch.Tensor:
        """
        Initialize the probabilities from prior distribution.
        """

        return self._prior_sampling(num_batch, num_res, device)

    def get_cond_vfs(
        self,
        trans_1: torch.Tensor,
        trans_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional vector fields for the Euclidean coordinates.

        Vector fields are obtained by taking time derivative of the optimal transport.
        The formula is given by:
        v(x, t) = (x_1 - x_t) / (1 - t)
        """

        vector_fields = (trans_1 - trans_t) / (1 - t)

        return vector_fields

    def sample(
        self,
        trans_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Sample noised Euclidean coordinates and vector fields at time t.

        trans_t is the interpolation of optimal transport between two gaussian distributions in R^3
        as described in the paper: https://arxiv.org/abs/2310.05297.

        The formula is given by:
        x_t = (1 - t) * x_0 + t * x_1
        """

        num_batch, num_res, _ = trans_1.shape

        trans_0 = self._prior_sampling(
            num_batch=num_batch, num_res=num_res, device=trans_1.device
        )

        trans_t = (1 - t) * trans_0 + t * trans_1

        vector_fields = self.get_cond_vfs(trans_1, trans_t, t)

        return trans_t, vector_fields


class CoordinateFlow(EuclideanFlow):
    """
    A flow matching framework for 3D coordinates.
    """

    def _prior_sampling(
        self,
        num_batch: int,
        num_res: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample from the centred gaussian distribution.
        """

        noise = torch.randn(num_batch, num_res, 3, device=device)

        return noise - torch.mean(noise, dim=-2, keepdims=True)


class SimplexFlow(EuclideanFlow):
    """
    A flow matching framework over the probability simplex.
    """

    def _prior_sampling(
        self,
        num_batch: int,
        num_res: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample from the uniform dirichlet distribution.
        """

        return torch.distributions.dirichlet.Dirichlet(
            torch.ones(20, device=device)
        ).sample((num_batch, num_res))


class SO3Flow:
    """
    A flow matching framework based on the rotation matching.
    """

    def __init__(self):
        pass

    def _uniform_so3_sample(
        self,
        num_batch: int,
        num_res: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample rotations from the uniform SO(3) distribution.
        """

        return torch.tensor(
            Rotation.random(num_batch * num_res).as_matrix(),
            device=device,
            dtype=torch.float32,
        ).reshape(num_batch, num_res, 3, 3)

    def init(self, num_batch: int, num_res: int, device: torch.device) -> torch.Tensor:
        """
        Initialize the orientations from the uniform SO(3) distribution.
        """

        return self._uniform_so3_sample(num_batch, num_res, device)

    def get_cond_vfs(
        self,
        rotmats_1: torch.Tensor,
        rotmats_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        vector_fields = calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)

        return vector_fields

    def sample(
        self,
        rotmats_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Sample noised rotations and vector fields at time t.

        rotmats_t is the geodesic interpolation of between SO(3) prior and data distributions.

        The formula is given by:
        R_t = Exp_{rotmats_0}(t * Log_{rotmats_0}(rotmats_1))
        """

        num_batch, num_res, _, _ = rotmats_1.shape

        rotmats_0 = self._uniform_so3_sample(
            num_batch=num_batch, num_res=num_res, device=rotmats_1.device
        )

        rotmats_t = geodesic_t(t, rotmats_1, rotmats_0)

        vector_fields = self.get_cond_vfs(rotmats_1, rotmats_t, t)

        return rotmats_t, vector_fields
