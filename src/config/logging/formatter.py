"""Format logging."""
from colorlog import ColoredFormatter

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
