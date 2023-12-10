"""Set up logging a base logger."""
import logging
import sys

from colorlog import ColoredFormatter, StreamHandler

from gesture_detection.config.base_config import get_base_config

# Set colors, style and format for logging.
logging_formatter = ColoredFormatter(
    fmt=(
        "%(white)s%(asctime)s%(reset)s | "
        "%(log_color)s%(levelname)s%(reset)s | "
        "%(blue)s%(filename)s:%(lineno)s%(reset)s | "
        "%(process)d >>> %(log_color)s%(message)s%(reset)s"
    ),
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

# Create logger.
base_logger = logging.getLogger("base_logger")

# Create stdout handler.
stdout = StreamHandler(stream=sys.stdout)

# Apply format to stdout handler.
stdout.setFormatter(logging_formatter)

# Set log level.
base_logger.setLevel(get_base_config().LOG_LEVEL)

# Add the handlers to the logger.
base_logger.addHandler(stdout)
