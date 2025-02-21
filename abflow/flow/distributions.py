"""
Author: Haowen Zhao
Email: hz362@cam.ac.uk

Implementations of manifold distributions.


Example usage of a concrete BaseDistribution class:

    >>> # Sampling
    >>> distribution = UniformSimplex(dim=20)
    >>> samples = distribution.sample(size=(10, 5), device=torch.device("cuda"))
"""

import torch
import torch.nn as nn
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from .rotation import rotvec_to_rotmat


class BaseDistribution(ABC):
    """
    Abstract base class for all distributions.
    All distributions must implement the `sample` method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Abstract method to sample from the distribution.

        :param size: batch shape of the tensors to sample.
        :param device: device to sample the tensor.
        :return: A sampled tensor from the distribution.
        """
        raise NotImplementedError


class EuclideanDistribution(BaseDistribution, ABC):
    """
    Abstract base class for distributions on the Euclidean space.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim


class NormalEuclidean(EuclideanDistribution):

    def __init__(self, dim: int):
        super().__init__(dim)

    def sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        samples = torch.randn(size + (self.dim,), device=device, dtype=dtype)
        return samples


class SimplexDistribution(BaseDistribution, ABC):
    """
    Abstract base class for distributions on the simplex space.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim


class UniformSimplex(SimplexDistribution):
    """
    Uniform distribution U[0, 1] on the simplex space.
    """

    def __init__(self, dim: int):
        super().__init__(dim)

    def sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:

        samples = torch.distributions.dirichlet.Dirichlet(
            torch.ones(self.dim, device=device, dtype=dtype)
        ).sample(size)

        return samples


class SO3Distribution(BaseDistribution, ABC):
    """
    Abstract base class for distributions on the SO(3) manifold.

    A point on the SO(3) manifold is a rotation vector [x, y, z]. Dimensionality is 3.
    rotation angle = ||[x, y, z]||.
    rotation axis = [x, y, z] / ||[x, y, z]||.
    """

    def __init__(
        self, var: float = 1.0, support_n: int = 5000, expansion_n: int = 5000
    ):

        super().__init__()

        self.var = var
        self.support_n = support_n
        self.expansion_n = expansion_n
        self.pi = math.pi

        self.support = np.linspace(0, math.pi, num=self.support_n + 1)[1:]
        self.densities = self.pdf(self.support)
        self.scores = self.score(self.support)
        self.cumulative_densities = np.cumsum(
            (self.support[1] - self.support[0]) * self.densities, axis=0
        )

    @abstractmethod
    def pdf(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """To be implemented by subclasses."""

        pass

    def cdf(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gives the cumulative density for some angle(s) under the parameterised IGSO3 distribution
        (see https://arxiv.org/pdf/2210.01776.pdf).
        """

        if isinstance(angle, float) or isinstance(angle, int):
            angle = np.array([angle], dtype=np.float64)

        densities = np.resize(
            self.densities[None, :], (angle.shape[0], self.densities.shape[0])
        )
        support = np.resize(
            self.support[None, :], (angle.shape[0], self.support.shape[0])
        )

        angle_support_index = np.argmin(np.abs(angle[:, None] - support), axis=-1)

        angle_support_values = support[np.arange(support.shape[0]), angle_support_index]
        zeroed_densities = np.where(
            support > angle_support_values[:, None], 0.0, densities
        )

        return np.trapz(zeroed_densities, x=support)

    def inv_cdf(self, cumulative_density: Union[float, np.ndarray]) -> np.ndarray:
        """
        Inverse of the cumulative density function, taking a cumulative density value as
        input and returning the correct value on the distribution's support.
        """
        assert np.all(cumulative_density >= 0), "The cumulative density must be >= 0"
        assert np.all(cumulative_density <= 1), "The cumulative density must be <1"

        return np.interp(cumulative_density, self.cumulative_densities, self.support)

    def score(self, angle: Union[float, np.ndarray], eps: float = 1e-12) -> np.ndarray:
        """
        Gets the gradient of the log PDF at a given angle or array of angles.
        Specifically this computes d log f(w)/dw via df(w)/dw * 1/f(w) (quotient rule).
        The argument `eps` is for numerical stability, and is added to the divisor.
        """
        if isinstance(angle, float) or isinstance(angle, int):
            angle = np.array([angle], dtype=np.float64)

        expansion_steps = np.arange(self.expansion_n)[None, :]
        a = expansion_steps + 0.5

        angle_expanded = angle[:, None]
        cos_half_angle = np.cos(angle_expanded / 2)
        cos_a_angle = np.cos(a * angle_expanded)
        sin_a_angle = np.sin(a * angle_expanded)
        sin_half_angle = np.sin(angle_expanded / 2)

        inf_sum_constant_terms = ((2 * expansion_steps) + 1) * np.exp(
            -expansion_steps * (expansion_steps + 1) * self.var
        )

        inf_sum = np.sum(
            inf_sum_constant_terms * (sin_a_angle / sin_half_angle),
            axis=-1,
        )
        inf_sum_derivative = np.sum(
            inf_sum_constant_terms
            * (
                ((a * cos_a_angle) / sin_half_angle)
                - ((cos_half_angle * sin_a_angle) / (2 * sin_half_angle**2))
            ),
            axis=-1,
        )

        return inf_sum_derivative / (inf_sum + eps)

    def sample_axis(self, size: torch.Size) -> torch.Tensor:
        """
        Sample a random unit vector as the rotation axis.
        """
        vectors = torch.randn(size + (3,))
        unit_vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

        return unit_vectors

    def sample_angle(self, size: torch.Size) -> np.ndarray:
        """
        Samples a random angle for rotation according to Eq. 5 in https://openreview.net/forum?id=BY88eBbkpe5
        """

        cdfs = np.random.rand(*size)
        angle = self.inv_cdf(cdfs)

        return angle

    def sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Samples one or more rotation matrices according to `size`, returned as a tensor of shape `size + (3, 3)`.
        """

        axes = torch.as_tensor(self.sample_axis(size), dtype=dtype, device=device)
        angles = torch.as_tensor(self.sample_angle(size), dtype=dtype, device=device)[
            ..., None
        ]

        rotvec = axes * angles
        rotmat = rotvec_to_rotmat(rotvec)

        return rotmat


class UniformSO3(SO3Distribution):

    def __init__(
        self, var: float = 1.0, support_n: int = 5000, expansion_n: int = 5000
    ):
        super().__init__(var, support_n, expansion_n)

    def pdf(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gives the probability density for some angle(s) under the parameterised IGSO3 distribution with some
        specified number of terms to expand the infinite series (see https://arxiv.org/pdf/2210.01776.pdf).
        """

        density = (1 - np.cos(angle)) / self.pi

        return density


class ToricDistribution(BaseDistribution, ABC):
    """
    Abstract base class for distributions on the toric space.
    """

    def __init__(self):
        super().__init__()


class UniformToric(ToricDistribution):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        x_0 = (
            torch.rand(size + (self.dim,), device=device, dtype=dtype) * 2 * math.pi
            - math.pi
        )
        return x_0
