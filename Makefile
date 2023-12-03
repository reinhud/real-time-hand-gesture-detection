POETRY := $(shell command -v poetry 2> /dev/null)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          Show this help"
	@echo "  clean         Clean the project"
	@echo "  test          Run the tests"
	@echo "  coverage      Run the tests with coverage"
	@echo "  lint          Run the linter"
	@echo "  format        Run the formatter"


.PHONY: clean
clean:
	find . -type d -name "__pycache__" | xargs rm -rf {}
	rm -rf .coverage .mypy_cache

.PHONY: test
test:
	poetry run pytest ./tests/

.PHONY: coverage
coverage: 
	poetry run pytest ./tests/ -ra --cov-report term-missing --cov=src

.PHONY: lint
lint:
	poetry run flake8 src tests

.PHONY: format
format:
	poetry run black src tests






