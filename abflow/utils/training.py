import torch

from ..model.model import AbFlow
from ..model.network import FlowPrediction
from ..model.datamodule import AntibodyAntigenDataModule


def setup_model(
    config: dict, checkpoint_path: str = None, load_optimizer: bool = False
):
    """Setup model and datamodule instances."""

    network_instance = FlowPrediction(**config["network"])
    datamodule_instance = AntibodyAntigenDataModule(config["datamodule"])

    if checkpoint_path is not None and load_optimizer:
        # Load full model state including optimizer
        model_instance = AbFlow.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            network=network_instance,
            **config["model"],
        )
    elif checkpoint_path is not None and not load_optimizer:
        # Load only model weights, ignore optimizer state
        checkpoint = torch.load(checkpoint_path)
        model_instance = AbFlow(network=network_instance, **config["model"])
        model_instance.load_state_dict(checkpoint["state_dict"])
    else:
        # start from scratch
        model_instance = AbFlow(network=network_instance, **config["model"])

    return model_instance, datamodule_instance
