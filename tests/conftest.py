"""Define general fixtures used throughout the test suite.

This file is automatically loaded by pytest before running any tests.
"""
from glob import glob


def refactor(string: str) -> str:
    """Refactor the string to be importable for pytest."""
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


# Allow fixtures that are in \tests\fixtures folder to be included in conftest by default.
pytest_plugins = [refactor(fixture) for fixture in glob("tests/fixtures/*.py") if "__" not in fixture]
