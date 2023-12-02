"""Set up logging."""
import logging
import sys

from colorlog import StreamHandler

from src.config.base_config import get_base_config
from src.config.logging.formatter import logging_formatter

# ========= Backend Logger Setup =========#
# Create logger.
logger = logging.getLogger("base_logger")

# Create stdout handler.
stdout = StreamHandler(stream=sys.stdout)

# Apply format to stdout handler.
stdout.setFormatter(logging_formatter)

# Set log level.
logger.setLevel(get_base_config().LOG_LEVEL)

# Add the handlers to the logger.
logger.addHandler(stdout)
