import psutil

from gesture_detection.config.base_logger import base_logger


def find_num_worker_default(config_value: int | None):
    """Find a reasonable default for the number of workers to use for data loading.

    The value is based on the number of CPU cores available.
    """
    if config_value is not None:
        return config_value
    else:
        base_logger.info("Finding default number of workers for data loading.")
        try:
            # CPU cores
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

            # Set workers as a percentage of available CPU cores
            default_workers = max(4, min(cpu_cores - 1, int(cpu_cores * 0.8)))
            base_logger.info(f"Available CPU cores: {cpu_cores}.")
            base_logger.info(f"Setting number of workers to: {default_workers} for data loading.")

            return default_workers

        except Exception as e:
            base_logger.info(f"Error while fetching system information: {e}")
            return 4
