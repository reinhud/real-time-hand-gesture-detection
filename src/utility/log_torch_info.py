import torch

from src.config.base_logger import base_logger


def log_torch_info() -> None:
    """Print info about Torch version and GPU availability."""
    base_logger.info("########## Torch Device Info ##########")

    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Create a list of GPU devices
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        # Display GPU information
        base_logger.info(f"{num_gpus} GPU(s) are available!")
        for i, device in enumerate(devices):
            base_logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        base_logger.info(f"Using GPUs: {', '.join([str(i) for i in range(num_gpus)])}")
    else:
        # If GPU is not available, fall back to CPU
        device = torch.device("cpu")
        base_logger.info("GPU is not available. Falling back to CPU.")

    # Display general PyTorch version and device information
    base_logger.info(f"PyTorch version: {torch.__version__}")
    base_logger.info(f"Using device: {device}")

    # Additional information if GPU is available
    if torch.cuda.is_available():
        base_logger.info(f"CUDA version: {torch.version.cuda}")
        base_logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")

    base_logger.info("########################################")


if __name__ == "__main__":
    log_torch_info()
