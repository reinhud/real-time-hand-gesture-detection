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

from gesture_detection.datasets.cifar10 import CIFAR10DataModule
from gesture_detection.models.resnet_test import LResNetTest
from gesture_detection.utility.style_training_output import model_summary, progress_bar

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


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [
        f.path for f in MlflowClient(tracking_uri="file:./mlflow_runs").list_artifacts(r.info.run_id, "model")
    ]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


if __name__ == "__main__":
    # MLFlow Setup
    mlflow.set_tracking_uri("file:./mlflow_runs")
    # Enable auto-logging to MLFlow to log metrics and parameters

    mlflow.pytorch.autolog()

    # Train the model.
    # with mlflow.start_run() as run:
    trainer.fit(model=MODEL, datamodule=DATAMODULE)

