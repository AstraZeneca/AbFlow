"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for training routine.

To use this script, run: python scripts/model/train.py -d sabdab 
The config file is at: ./config/sabdab.yaml
"""

import os
import torch
import yaml
import traceback

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from datetime import timedelta

from abflow.utils.training import setup_model
from abflow.utils.arguments import get_arguments, get_config, print_config_summary

# Set the NCCL blocking wait environment variable
os.environ["NCCL_BLOCKING_WAIT"] = "1"


def train(config: dict):
    """Function for training and saving the model."""
    # Save the config file to keep a record of the settings
    results_dir = (
        f"{config['datamodule']['dataset']['paths']['model']}/{config['model_name']}"
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_dir + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    # Instantiate model and datamodule
    model, datamodule = setup_model(
        config, config["checkpoint"]["path"], config["checkpoint"]["load_optimizer"]
    )

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
    logger = WandbLogger(
        project=config["project_name"], name=config["model_name"], log_model=False
    )

    # Trainer
    trainer = Trainer(
        devices=config["trainer"]["devices"],
        accelerator="gpu",
        strategy=DDPStrategy(
            timeout=timedelta(seconds=15400), find_unused_parameters=True
        ),
        precision=config["trainer"]["precision"],
        max_epochs=config["trainer"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=config["trainer"]["val_check_interval"],
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    print("Done with training...")


def main(config: dict):
    """Main wrapper function for training routine.

    :param config: Dictionary containing options and arguments.
    """
    train(config)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Summary of the config
    print_config_summary(config, args)

    # ----- Run Training
    try:
        main(config)
    except Exception as e:
        print(traceback.format_exc())
