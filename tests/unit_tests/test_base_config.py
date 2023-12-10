import pytest
from pydantic_settings import BaseSettings

from gesture_detection.config.base_config import get_base_config


class TestBaseConfig:
    """Tests for the BaseConfig class."""

    def test_correct_config_loaded_from_env_file(self, monkeypatch):
        """Ensure the correct config is loaded."""
        # Arrange
        monkeypatch.setenv("BASE_ENVIRONMENT", "development")
        monkeypatch.setenv("BASE_LOG_LEVEL", "DEBUG")

        # Assert
        assert get_base_config().ENVIRONMENT == "development"
        assert get_base_config().LOG_LEVEL == "DEBUG"

    def test_only_env_variables_with_prefix_loaded(self, monkeypatch):
        """Ensure only env variables with "BASE" prefix are loaded."""
        # Arrange
        monkeypatch.setenv("BASE_ENVIRONMENT", "development")
        monkeypatch.setenv("BASE_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOG_LEVEL", "CRITICAL")

        # Assert
        assert get_base_config().ENVIRONMENT == "development"
        assert get_base_config().LOG_LEVEL == "DEBUG"

    def test_case_insetivity_of_env_variable_names(self, monkeypatch):
        """Ensure env variable names are case insensitive."""
        environment_value = "development"
        log_level_value = "DEBUG"

        # Arrange
        monkeypatch.setenv("base_environment", environment_value)
        monkeypatch.setenv("base_log_level", log_level_value)

        lowercase_environment = get_base_config().ENVIRONMENT
        lowercase_log_level = get_base_config().LOG_LEVEL

        monkeypatch.setenv("BASE_ENVIRONMENT", environment_value)
        monkeypatch.setenv("BASE_LOG_LEVEL", log_level_value)

        uppercase_environment = get_base_config().ENVIRONMENT
        uppercase_log_level = get_base_config().LOG_LEVEL

        # Assert
        assert lowercase_environment == uppercase_environment
        assert lowercase_log_level == uppercase_log_level

    def test_validate_env_variables(self, monkeypatch):
        """Ensure env variables are validated."""
        # Arrange
        monkeypatch.setenv("BASE_ENVIRONMENT", "foo")
        monkeypatch.setenv("BASE_LOG_LEVEL", "bar")

        # Assert
        with pytest.raises(Exception) as e_info:
            # try to access env variable that has invalid value
            assert get_base_config().ENVIRONMENT == "foo"
            print(e_info)

    def test_get_base_config(self):
        """Ensure the get_base_config method returns the BaseConfig."""
        # Arrange
        base_config = get_base_config()

        # Assert
        assert isinstance(base_config, BaseSettings)
