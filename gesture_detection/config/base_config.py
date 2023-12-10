"""Define Configs used throughout the project."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """Base Config for the project.

    This is using pydantic_settings to load configs from env file and validate them.
    """

    # Configure based on env file
    model_config = SettingsConfigDict(
        env_prefix="BASE_",
        env_file="/workspaces/real-time-hand-gesture-detection/.env",
        env_file_encoding="utf-8",
    )

    # Project settings
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    USE_GPU: bool = False

    # MLFlow settings
    MLFLOW_TRACKING_URI: str = "file:./mlflow_runs"

    # Data settings
    DATA_DIR: str = "/workspaces/real-time-hand-gesture-detection/data"


@lru_cache()  # Cache the settings.
def get_base_config() -> BaseSettings:
    """Provide the BaseConfig."""
    return BaseConfig()
