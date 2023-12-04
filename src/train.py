"""Run this file to train the model."""
import lightning as L
import mlflow
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import PyTorchProfiler
from mlflow import MlflowClient

from src.data_modules.cifar10 import CIFAR10DataModule
from src.models.resnet_test import LResNet_test
from src.utility.style_training_output import model_summary, progress_bar

# ================================================== #
# Set training config                                #
# ================================================== #
MODEL = LResNet_test()
DATAMODULE = CIFAR10DataModule()

MAX_EPOCH = 2
NUM_SANITY_VAL_STEPS = 2

# For logging MLFlow
EXPERIMENT_NAME = "test"
# ================================================== #
# ================================================== #

# Monitor pytorch functions to find bottlenecks
profiler = PyTorchProfiler()

# Set up logger for MLFlow
mlf_logger = MLFlowLogger(
    experiment_name=EXPERIMENT_NAME,
    tracking_uri="file:./mlflow_runs",
    log_model=True,
)

# Set up trainer
trainer = L.Trainer(
    precision="16-mixed",
    # fast_dev_run=True,
    max_epochs=MAX_EPOCH,
    num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
    logger=mlf_logger,
    # profiler=profiler,
    callbacks=[DeviceStatsMonitor(), progress_bar, model_summary],
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
    # Enable auto-logging to MLFlow to log metrics and parameters
    # MLFlow Setup
    mlflow.set_tracking_uri("file:./mlflow_runs")
    # mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog()

    # Train the model.
    # with mlflow.start_run() as run:
    trainer.fit(model=MODEL, datamodule=DATAMODULE)

    # Fetch the auto logged parameters and metrics.
    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
