"""Run this file to train the model."""
from typing import Literal

import lightning as L
import mlflow
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import MLFlowLogger

from src.data_modules.cifar10 import CIFAR10DataModule
from src.models.resnet_test import LResNetTest
from src.utility.style_training_output import model_summary, progress_bar

# ================================================== #
# Set training config                                #
# ================================================== #
MODEL = LResNetTest()
DATAMODULE = CIFAR10DataModule()

MAX_EPOCH = 2
NUM_SANITY_VAL_STEPS = 2

# For logging MLFlow
EXPERIMENT_NAME = "Testing MLFlow"
DATASET_NAME: Literal["CIFAR10", "nvGesture", "egoGesture"] = "CIFAR10"
# ================================================== #
# ================================================== #
mlflow_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tracking_uri="file:./mlflow_runs",
    log_model=True,
    tags={"dataset": DATASET_NAME, "model": MODEL.__class__.__name__},
)

# Set up trainer
callbacks = [
    # StochasticWeightAveraging(swa_lrs=1e-2),
    DeviceStatsMonitor(),
    ModelSummary(max_depth=-1),
    ModelCheckpoint(monitor="val_loss", mode="min"),
    progress_bar,
    model_summary,
]
trainer = L.Trainer(
    precision="16-mixed",
    # fast_dev_run=True,
    max_epochs=MAX_EPOCH,
    num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
    profiler="simple",
    # logger=mlflow_logger,
    callbacks=callbacks,
)


if __name__ == "__main__":
    # MLFlow Setup
    mlflow.set_tracking_uri("file:./mlflow_runs")
    # Enable auto-logging to MLFlow to log metrics and parameters
    mlflow.pytorch.autolog()

    # Train the model.
    # with mlflow.start_run() as run:
    trainer.fit(model=MODEL, datamodule=DATAMODULE)
