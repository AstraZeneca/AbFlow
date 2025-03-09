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


from datetime import timedelta
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from abflow.utils.training import setup_model, set_seed
from abflow.utils.arguments import get_arguments, get_config, print_config_summary
from torch.profiler import ProfilerActivity
from pytorch_lightning.profilers import PyTorchProfiler



# Define a short schedule (adjust wait/warmup/active as needed)
schedule = torch.profiler.schedule(
    wait=0,  # Number of steps to skip profiling
    warmup=1,  # Number of steps to warm up before collecting
    active=3,  # Number of steps to actively profile
    repeat=1,
)

profiler = PyTorchProfiler(
    dirpath=".",  # Directory to save profiling traces
    filename="profile_trace",  # File prefix for traces
    activities=[  # What to profile (CPU and/or GPU)
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    record_shapes=True,  # Include shapes of operator inputs
    profile_memory=True,  # Track memory usage
    with_stack=True,  # Collect stack traces
)

def train(config: dict):
    """Function for training and saving the model."""
    # Save the config file to keep a record of the settings
    results_dir = (
        f"{config['datamodule']['dataset']['paths']['model']}/{config['model_name']}_{config['datamodule']['dataset']['name']}_{'_'.join(config['shared']['design_mode'])}"
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
    if config['datamodule']['dataset']['name'] == 'sabdab':
        checkpoint_callback = ModelCheckpoint(
            dirpath=results_dir,
            filename="{epoch:02d}",
            save_top_k=-1,
            verbose=True,
            every_n_epochs=5,
            monitor=None,
        )

    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=results_dir,
            filename="checkpoint-{step}",
            save_top_k=-1,
            every_n_train_steps=2000,
        )

    callbacks.append(checkpoint_callback)

    # Progress bar
    tqdm = TQDMProgressBar(refresh_rate=10)
    callbacks.append(tqdm)

    # Logger
    logger = WandbLogger(
        project=config["project_name"], name=config["model_name"], log_model=False
    )

    # Trainer
    trainer = L.Trainer(
        devices=8,
        accelerator="gpu",
        strategy=DDPStrategy(
            timeout=timedelta(seconds=15400), find_unused_parameters=True,
        ),
        precision='bf16',
        max_epochs=config["trainer"]["max_epochs"],
        accumulate_grad_batches=config["trainer"]["accumulate_grad_batches"],
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=config["trainer"]["val_check_interval"],
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        # profiler=profiler,
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
    # Set the seed
    set_seed(config['shared']['seed'])

    # ----- Run Training
    try:
        main(config)
    except Exception as e:
        print(traceback.format_exc())
