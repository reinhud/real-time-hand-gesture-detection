import mlflow
from lightning.pytorch.cli import LightningCLI
from rich import get_console

from src.callbacks.model_summary import CustomRichModelSummary
from src.callbacks.progress_bar import CustomRichProgressBar
from src.callbacks.run_printer import RunPrinter

# 🚨❗Copy all lightning models and data modules here to be registered by LightningCLI
from src.datasets.cifar10 import CIFAR10DataModule  # noqa: F401
from src.models.resnet_test import LResNetTest  # noqa: F401


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--mlflow_experiment", default="Default")

    def before_fit(self):
        # print(self.arg.notification_email)
        if self.trainer.global_rank == 0:
            mlflow.set_tracking_uri("file:./mlflow_runs")
            # TODO: why is this not working
            # mlflow.set_experiment(self.config["mlflow_experiment"])
            mlflow.set_experiment("MLFlow Test")
            mlflow.pytorch.autolog()
            # TODO: make mlflow tags automatic from trainer and model
            mlflow.set_tags({"data": "CIFAR10", "Model": "LResNetTest"})


def cli_main():
    """Create a CLI to train models flexibly.

    This sets up a Trainer instance when involked.

    Methods:
        fit:         Runs the full optimization routine.
        validate:    Perform one evaluation epoch over the validation set.
        test:        Perform one evaluation epoch over the test set.
        predict:     Run inference on your data.

    The cli can be involved in two ways:
    1. Directly call this file with
        `python src/cli.py fit --trainer.max_epochs=10 --model=LResNetTest --data=CIFAR10DataModule"`
    2. From the root directory
        `make cli ARGS="fit --trainer.max_epochs=10 --model=LResNetTest --data=CIFAR10DataModule"`

    Prefered Usage🚀:
    The cli can also accept arguments from a config file.
    See https://pytorch-lightning.readthedocs.io/en/latest/common/config.html#config-file.
    This makes it also easy to override the config file with command line arguments
    and test models quickly without having to change our implementation.
    Basic Pytorch Optimizer and Scheduler can be used automatically by specifying them in the.
    🔥Please create a config file for each model to make it easier to play around experiment with .💦

    To integrate easily with MLFlow, please use the following options:
    --mlflow_experiment="Testing MLFlow"

    Examples:
        make cli ARGS="fit --config src/config/model_config/resnet_test.yaml"
        make cli ARGS="fit --config src/config/model_config/resnet_test.yaml --mlflow_experiment='Testing MLFlow'"    # noqa: E501
        make cli ARGS="fit --config src/config/model_config/resnet_test.yaml --trainer.max_epochs=10 --lr=1e-2"     # noqa: E501
        make cli ARGS="fit --model Model1 --data FakeDataset1 --optimizer LitAdam --lr_scheduler LitLRScheduler"    # noqa: E501
    """
    # Indicate run start in console
    console = get_console()
    console.rule(style="purple")
    console.rule("[bold green_yellow]STARTED NEW RUN", style="purple")
    console.rule(style="purple")

    cli = CustomLightningCLI(  # noqa: F841 pylint: disable=unused-variable
        # Set default callbacks that cant be overriden from config file
        trainer_defaults={
            "num_sanity_val_steps": 2,
            "callbacks": [
                CustomRichProgressBar(),
                CustomRichModelSummary(max_depth=-1),
                RunPrinter(),
            ],
        },
    )

    # Indicate run end in console
    console = get_console()
    console.rule(style="purple")
    console.rule("[bold green_yellow]FINISHED RUN", style="purple")
    console.rule(style="purple")



if __name__ == "__main__":
    cli_main()
