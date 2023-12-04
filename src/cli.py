from lightning.pytorch.cli import LightningCLI

import src.data_modules.cifar10  # noqa: F401
import src.models.resnet_test  # noqa: F401

# import src.optimizers  # noqa: F401


def cli_main():
    """Create a CLI to train models flexibly."""
    cli = LightningCLI()  # noqa: F841 pylint: disable=unused-variable


if __name__ == "__main__":
    cli_main()
