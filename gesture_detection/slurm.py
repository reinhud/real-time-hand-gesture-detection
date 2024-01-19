import os

from lightning.pytorch.cli import LightningCLI


def main():
    slurm_id = os.environ.get("SLURM_ID", 0)
    print(f"SLURM ID: {slurm_id}")
    root_config_file = "gesture_detection/config/model_config/lstm.yaml"
    specific_config_file = f"gesture_detection/config/model_config/slurm/{slurm_id}.yaml"
    cli = LightningCLI(
        args=[
            "fit",
            "--config", root_config_file,
            "--config", specific_config_file
        ]
    )


if __name__ == "__main__":
    main()
