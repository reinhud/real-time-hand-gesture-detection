import torch


def log_pytorch_info() -> None:
    """Print info about PyTorch version and GPU availability."""
    print("########## PyTorch Device Info ##########")

    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Create a list of GPU devices
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        # Display GPU information
        print(f"{num_gpus} GPU(s) are available!")
        for i, device in enumerate(devices):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        print(f"Using GPUs: {', '.join([str(i) for i in range(num_gpus)])}")
    else:
        # If GPU is not available, fall back to CPU
        device = torch.device("cpu")
        print("GPU is not available. Falling back to CPU.")

    # Display general PyTorch version and device information
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    # Additional information if GPU is available
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")


if __name__ == "__main__":
    log_pytorch_info()
