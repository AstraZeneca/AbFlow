"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for training routine.
"""

import os
import copy
import time
import torch.distributed as dist
import torch

from datetime import date, timedelta

import yaml
import traceback
from utils.load_data import Loader
from utils.arguments import get_arguments, get_config, print_config_summary
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from abflow.model.model import AbFlow
from abflow.model.network import FlowPrediction

# Set the NCCL blocking wait environment variable
os.environ["NCCL_BLOCKING_WAIT"] = "1"


def setup_model(config, checkpoint_path: str = None, load_optimizer=False):

    network_instance = FlowPrediction(**config["network"])

    if checkpoint_path is not None and load_optimizer:
        # Load full model state including optimizer
        model_instance = AbFlow.load_from_checkpoint(
            checkpoint_path=checkpoint_path, network=network_instance, **config["model"]
        )
    elif checkpoint_path is not None and not load_optimizer:
        # Load only model weights, ignore optimizer state
        checkpoint = torch.load(checkpoint_path)
        model_instance = AbFlow(network=network_instance, **config["model"])
        model_instance.load_state_dict(checkpoint["state_dict"])
    else:
        # start from scratch
        model_instance = AbFlow(network=network_instance, **config["model"])

    return model_instance


def train(config, data_loader, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # Save the config file to keep a record of the settings
    results_dir = f"{config['paths']['model']}/{config['model_name']}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_dir + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    # Instantiate model
    model = setup_model(config)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=results_dir,
        filename="{epoch:02d}",
        save_top_k=1,
        verbose=True,
        every_n_epochs=1,
        monitor=None,
    )
    callbacks.append(checkpoint_callback)

    # Logger
    today = date.today()
    logger = WandbLogger(
        project=config["project_name"], name=config["model_name"], log_model=False
    )

    # Get data loaders
    train_loader = data_loader.train_loader
    validation_loader = data_loader.validation_loader
    test_loader = data_loader.test_loader

    # Fit the model to the data
    trainer = L.Trainer(
        devices=config["trainer"]["devices"],
        accelerator="gpu",
        strategy=DDPStrategy(
            timeout=timedelta(seconds=15400), find_unused_parameters=True
        ),
        precision=32,
        max_epochs=config["trainer"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=config["trainer"]["val_check_interval"],
        log_every_n_steps=10,
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
    )

    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, test_loader)
    print("Done with training...")


def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.
    """
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"])

    # Start training and save model weights at the end
    train(config, ds_loader, save_weights=True)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Define the name of the model --- which will be used for saving the model as well as for tracking the results in W&B
    # config["model_name"] = "test_100_epochs_4gpu"
    # Summary of the config
    print_config_summary(config, args)

    # ----- Run Training
    try:
        main(config)
    except Exception as e:
        print(traceback.format_exc())
