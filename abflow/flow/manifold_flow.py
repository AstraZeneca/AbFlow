"""
Author: Haowen Zhao
Email: hz362@cam.ac.uk

Implementations of flow-based generative frameworks for manifold learning.

Example usage of a concrete ManifoldFlow class:

    >>> # Training
    >>> flow = OptimalTransportEuclideanFlow(dim=10)
    >>> x_1 = torch.randn(1, 10) # True data point
    >>> t = torch.tensor(0.5) # sampled time step
    >>> x_t = flow.interpolate_path(x_1, t)
    >>> v_t = flow.get_cond_vfs(x_t, x_1, t)
    >>> if neural network predicts x_1_hat:
        >>> x_1_hat = neural_network(x_t)
        >>> x_1_hat = flow.nn_to_manifold(x_1_hat)
        >>> v_t_hat = flow.get_cond_vfs(x_t, x_1_hat, t)
    >>> if neural network predicts v_t_hat:
        >>> v_t_hat = neural_network(x_t)
    >>> loss = loss_fn(v_t, v_t_hat)


    >>> # Inference
    >>> x_0 = flow.prior_sample(size=(1, 10), device=device)
    >>> x_t = x_0
    >>> for t in torch.linspace(0, 1, 10):
    >>>     x_1_hat = neural_network(x_t)
    >>>     x_1_hat = flow.nn_to_manifold(x_1_hat)
    >>>     v_t = flow.get_cond_vfs(x_t, x_1_hat, t)
    >>>     x_t = flow.update_x(x_t, v_t, dt=0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from einops import rearrange
from .scheduler import FlowScheduleTypes, get_flow_schedule
from .distributions import (
    BaseDistribution,
    NormalEuclidean,
    UniformSimplex,
    UniformSO3,
    UniformToric,
)
from .rotation import (
    rotvecs_mul,
    rotvec_inv,
    rotvec_to_rotmat,
    rotmat_to_rotvec,
    rotmat_to_rot6d,
    rotvec_to_rotmat,
    rotmat_inv,
    rotmats_mul,
)


class ManifoldFlow(ABC):
    """
    Base abstract class for manifold flow frameworks.

    Child classes **must** instantiate the following attributes:
        - _schedule: The flow schedule with FlowSchedule class.
        - _prior: The prior distribution with BaseDistribution class.

    Child classes **must** implement the following 3 core methods:
        - nn_to_manifold: Restricts neural network outputs to the manifold.
        - log_map: Computes the logarithmic map between two points on the manifold / transformed space.
        - exp_map: Computes the exponential map to move along the manifold.
    """

    def __init__(self):

        super().__init__()

    def prior_sample(
        self, size: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Sample from the prior distribution.

        :param size: The batch shape of the tensors to sample.
        :param device: The device to put the sampled tensor on.
        :param dtype: The data type of the sampled tensor.
        :return: The sampled point from the prior distribution.
        """

        return self._prior.sample(size, device, dtype)

    def interpolate_path(self, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between prior x_0 and true x_1 on the manifold over the time
        interval [0, 1].

        :param x_1: The ending point.
        :param t: The interpolation time.
        :return: The interpolated point.
        """
        x_0 = self.prior_sample(x_1.size()[:2], x_1.device, x_1.dtype)
        alpha_t, beta_t, _ = self._schedule(t)
        x_t = self.exp_map(x_0, beta_t * self.log_map(x_0, x_1))

        return x_t

    def get_cond_vfs(
        self, x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Conditional vector fields (defined directly on manifold or on transformed space)
        at time t.

        :param x_t: The current point on the manifold.
        :param x_1: The ending point.
        :param t: The interpolation time.
        :return: The conditional vector field at time t.
        """
        _, _, speed_t = self._schedule(t)
        v_t = speed_t * self.log_map(x_t, x_1)

        return v_t

    def update_x(
        self, x_t: torch.Tensor, v_t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Update the point x_t on the manifold using the vector field v_t and time step dt.

        The update is computed using the exponential map to move along the manifold
        in the direction of the vector field v_t scaled by dt. This method approximates
        the solution of the differential equation using Euler's method for manifolds.

        Formula:
            x_t = exp_map(x_t, v_t * dt)

        :param x_t: The current point on the manifold.
        :param v_t: The vector field at time t.
        :param dt: The time step for the update.
        :return: The updated point on the manifold after time step dt.
        """
        x_updated = self.exp_map(x_t, v_t * dt)
        return x_updated

    @abstractmethod
    def nn_to_manifold(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Restrict neural network unconstrained outputs to the manifold.
        """
        raise NotImplementedError


class EuclideanFlow(ManifoldFlow, ABC):
    """
    Abstract class for flow on Euclidean space.
    """

    def log_map(self, x_base: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:

        return x_target - x_base

    def exp_map(self, x_base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        return x_base + v

    def nn_to_manifold(self, x_hat: torch.Tensor):
        return x_hat

    # def nn_to_manifold(self, x_hat: torch.Tensor) -> torch.Tensor:
    #     """Clamp translation vectors"""
    #     return torch.clamp(x_hat, -100.0, 100.0)


class OptimalTransportEuclideanFlow(EuclideanFlow):
    """
    Optimal transport flow path in R^n. The Linear schedule here is flow matching in Lipman et al paper
    and 1-st rectified flow in Liu et al paper if reflow is not used. Note improving inference speed is
    not the focus and generation quality were found to degrade for k-th rectified flow with
    k > 1 (Table 1 in Liu et al paper).

    paper: Flow Matching for Generative Modelling, ICLR, Lipman et al.
    link: https://arxiv.org/abs/2210.02747
    paper: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified
            Flow, ICLR, Liu et al.
    link: https://arxiv.org/abs/2209.03003
    """

    def __init__(
        self,
        dim: int,
        schedule_type: FlowScheduleTypes = "linear",
        schedule_params: dict = {},
    ):
        """
        Initialize the flow with a given schedule type.

        :param dim: The data dimensionality.
        :param schedule_type: The type of flow schedule to use (linear, exp_decay, inv_exp_growth).
        :param schedule_params: The parameters for the flow schedule.
        """

        super().__init__()

        self._schedule = get_flow_schedule(schedule_type, schedule_params)
        self._prior = NormalEuclidean(dim)


class SimplexFlow(ManifoldFlow, ABC):
    """
    Abstract class for flow on the simplex manifold.
    """

    def nn_to_manifold(self, x_hat: torch.Tensor, temp: float = 0.10):
        """
        Neural network output a vector of dimension n from [-inf, inf] to [0, 1] by
        applying the softmax function.
        """

        return torch.softmax(x_hat / temp, dim=-1)

    # def get_cond_vfs(
    #     self, x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Conditional vector fields (defined directly on manifold or on transformed space)
    #     at time t.

    #     :param x_t: The current point on the manifold.
    #     :param x_1: The ending point.
    #     :param t: The interpolation time.
    #     :return: The conditional vector field at time t.
    #     """
    #     _, _, speed_t = self._schedule(t)
    #     v_t = speed_t * self.log_map(x_t, x_1)

    #     temp_mask = x_t == x_1
    #     print(f"Seqeuence get cond vfs: speed_t: {speed_t[:20]}, x_t: {x_t[temp_mask][0]}, x_1: {x_1[temp_mask][0]}")

    #     return v_t


class LinearSimplexFlow(SimplexFlow):
    """
    Linear interpolation flow directly on the simplex manifold, representing the simplex
    as a subspace of an (n-1)-dimensional Euclidean space and following the optimal
    transport path.
    """

    def __init__(
        self,
        dim: int,
        schedule_type: FlowScheduleTypes = "linear",
        schedule_params: dict = {},
    ):

        super().__init__()

        self._schedule = get_flow_schedule(schedule_type, schedule_params)
        self._prior = UniformSimplex(dim)

    def log_map(self, x_base: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Logarithmic map on the euclidean space."""

        return x_target - x_base

    def exp_map(self, x_base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on the euclidean space."""

        return x_base + v

    def tangent_project(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project a vector v onto the tangent space of the simplex.

        The tangent space of the simplex consists of all vectors whose components sum to zero.
        This projection is achieved by subtracting the mean of v from each component, which
        guarantees that the resulting vector has a zero sum.

        Args:
            v (torch.Tensor): The vector to be projected.

        Returns:
            torch.Tensor: The projected vector with components that sum to zero.
        """
        return v - v.mean(dim=-1, keepdim=True)


class SO3Flow(ManifoldFlow, ABC):
    """
    Flow for the SO(3) manifold. A point x on the SO(3) manifold is a [..., 3]
    rotation vector (axis-angle) representation in this implementation.
    """

    # def nn_to_manifold(self, x_hat: torch.Tensor):
    #     """
    #     neural network output a 3D vector for the rotation vector on the SO(3) manifold.
    #     """

    #     return x_hat

    def nn_to_manifold(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Ensure valid rotation vectors"""
        return x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)


class LinearSO3Flow(SO3Flow):
    def __init__(self, schedule_type="linear", schedule_params=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self._schedule = get_flow_schedule(schedule_type, schedule_params or {})
        self._prior = UniformSO3()

    def prior_sample(self, size, device, dtype):
        """Sample rotations with correct spatial dimensions"""
        # Size should be (batch, seq_len)
        rot_mat = self._prior.sample(size=size, device=device, dtype=dtype)
        return rotmat_to_rot6d(rot_mat)  # Converts [..., 3, 3] -> [..., 6]

    def log_map(
        self, x_base: torch.Tensor, x_target: torch.Tensor, is_6d: bool = True
    ) -> torch.Tensor:
        """Direct vector difference in 6D space"""
        # 6D rotation
        if is_6d:
            return x_target - x_base  # Simple Euclidean difference
        # 3D rotation
        else:
            x_base_inv = rotmat_inv(x_base)
            R_rel = rotmats_mul(x_base_inv, x_target)
            v = rotmat_to_rotvec(R_rel)
            return v

    def exp_map(
        self, x_base: torch.Tensor, v: torch.Tensor, is_6d: bool = True
    ) -> torch.Tensor:
        """Linear update in 6D space"""
        # 6D rotation
        if is_6d:
            return x_base + v
        # 3D rotation
        else:
            x_target = rotmats_mul(x_base, rotvec_to_rotmat(v))
            return x_target

    def nn_to_manifold(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Project raw network output to valid 6D rotation representation.
        Args:
            x_hat: Raw network output [..., 6]

        Returns:
            Valid 6D rotation representation [..., 6]
        """
        a1 = x_hat[..., :3]
        a2 = x_hat[..., 3:]

        e1 = F.normalize(a1, dim=-1)
        u2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(u2, dim=-1)

        return torch.cat([e1, e2], dim=-1)

    def interpolate_path(self, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between prior x_0 and true x_1 on the manifold over the time
        interval [0, 1].

        :param x_1: The ending point.
        :param t: The interpolation time.
        :return: The interpolated point.
        """
        # x_1 shape: [6, 240, 6]
        # t shape: [6, 240, 1]

        # Sample prior with matching spatial dimensions
        x_0 = self.prior_sample(x_1.shape[:-1], x_1.device, x_1.dtype)  # [6, 240, 6]
        # Get schedule parameters
        _, beta_t, _ = self._schedule(t)  # beta_t shape [6, 240, 1]

        # Linear interpolation in 6D space
        return self.exp_map(x_0, beta_t * self.log_map(x_0, x_1))

    # def nn_to_manifold(self, x_hat: torch.Tensor):
    #     """No conversion needed - network outputs 6D directly"""
    #     return x_hat


class ToricFlow(ManifoldFlow, ABC):
    """
    Flow for the toric manifold.
    """

    def nn_to_manifold(self, x_hat: torch.Tensor):
        """
        Neural network output a vector of dimension n from [-inf, inf] to [-pi, pi].
        """

        return self.wrap_angle(x_hat)


class LinearToricFlow(ToricFlow):
    """
    Linear interpolation flow for the toric manifold.

    paper: Full-Atom Peptide Design based on Multi-modal Flow Matching
    link: https://arxiv.org/abs/2406.00735
    """

    def __init__(
        self,
        dim: int,
        schedule_type: FlowScheduleTypes = "linear",
        schedule_params: dict = {},
    ):

        super().__init__()

        self._schedule = get_flow_schedule(schedule_type, schedule_params)
        self._prior = UniformToric(dim)

    def wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrap the angles from [-inf, inf] to [-pi, pi].
        """

        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

    def log_map(self, x_base: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the vector field Log_{x_base}(x_target).

        Formula:
            log_{x_base}(x_target) = arctan2(sin(x_target - x_base), cos(x_target - x_base))
                                   = wrap_angles(x_target - x_base)
        """

        return self.wrap_angle(x_target - x_base)

    def exp_map(self, x_base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the exponential map Exp_{x_base}(v).

        Formula:
            exp_{x_base}(v) = wrap_angles(x_base + v)
        """

        return self.wrap_angle(x_base + v)
