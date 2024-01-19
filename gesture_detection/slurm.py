import os

from lightning.pytorch.cli import LightningCLI


def main():
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")
    slurm_id = os.environ.get("SLURM_LOCALID", 0)
    print(f"SLURM ID: {slurm_id}")
    root_config_file = "gesture_detection/config/model_config/lstm.yaml"
    specific_config_file = f"gesture_detection/config/model_config/slurm/{slurm_id}.yaml"
    cli = LightningCLI(
        args=[
            "fit",
            "--config", root_config_file,
            "--config", specific_config_file,
            "--trainer.logger", "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
            "--trainer.logger.init_args.save_dir", "lightning_logs",
            "--trainer.logger.init_args.name", f"job_{slurm_job_id}",
            "--trainer.logger.init_args.version", f"slurm_{slurm_id}",
        ]
    )


if __name__ == "__main__":
    main()
