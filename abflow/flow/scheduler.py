import torch
from abc import ABC, abstractmethod
from typing import Tuple, Literal, Dict

FlowScheduleTypes = Literal["linear"]


class BaseSchedule(ABC):
    """
    Abstract base class for flow schedules.

    Calculate the interpolation and speed parameters for the given schedule type and time `t`.
    Exponential scale `c` is set to 5.0.

    This function is used for controlling linear interpolation between two points in any space.
    However, it is not appropriate for cases where the path is nonlinear or when the interpolation
    process is not deterministic (e.g., diffusion paths).
    """

    @abstractmethod
    def forward(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to compute alpha_t, beta_t, and speed_t.

        :param t: The time step as a tensor, clamped to avoid division by zero at t = 1.
        :return: alpha_t, the interpolation coefficient for the prior x_0,
                    beta_t, the interpolation coefficient for the true x_1,
                    speed_t, the speed at time t, dictating the rate of change.
        """
        pass

    def __call__(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Allows instances of the class to be called like a function.
        Simply calls the forward method.
        """
        return self.forward(t)


class LinearSchedule(BaseSchedule):

    def forward(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Formula:
            alpha_t = 1 - t,
            beta_t = t,
            speed_t = 1 / (1 - t)
        """
        alpha_t = 1 - t
        beta_t = t
        speed_t = 1 / (1 - t)
        return alpha_t, beta_t, speed_t


def get_flow_schedule(
    schedule_type: FlowScheduleTypes, schedule_params: Dict[str, float] = {}
) -> BaseSchedule:
    """
    Factory method to return the appropriate schedule class based on type.

    :param schedule_type: Type of the schedule ('linear').
    :param schedule_params: A dictionary of parameters used for schedule initialization (if needed).
    :return: An instance of a subclass of `BaseSchedule`.
    """

    if schedule_type == "linear":
        return LinearSchedule()

    else:
        raise ValueError(
            f"Invalid schedule type {schedule_type}. Should be one of {FlowScheduleTypes.__args__}."
        )
