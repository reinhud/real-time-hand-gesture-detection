.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"

	@echo "  help          		    		Show this help"
	@echo "  torch-info   					Show torch information related to the project"
	@echo "  clean-code       				Clean the project"
	@echo "  test-code          			Run the tests"
	@echo "  coverage      					Run the tests with coverage"
	@echo "  lint-code          			Run the linter"
	@echo "  format-code        			Run the formatter"
	@echo "  cli ARGS="<command> --<arg>"	Use Lightning CLI"
	@echo "  cli-help        				Show Lightning CLI help"

.PHONY: torch-info
torch-info:
	poetry run python src/utility/log_torch_info.py


.PHONY: clean-code
clean-code:
	find . -type d -name "__pycache__" | xargs rm -rf {}
	rm -rf .coverage .mypy_cache

.PHONY: test-code
test-code:
	poetry run pytest ./tests/

.PHONY: coverage
coverage: 
	poetry run pytest ./tests/ -ra --cov-report term-missing --cov=src

.PHONY: lint-code
lint-code:
	poetry run flake8 src

.PHONY: format-code
format-code:
	poetry run isort src
	poetry run black src

.PHONY: cli
cli:
	poetry run python src/cli.py $(ARGS)

.PHONY: cli-help
cli-help:
	poetry run python src/cli.py --help

.PHONY: ui
ui:
	poetry run mlflow ui --backend-store-uri mlflow_runs 

.PHONY: train
train:
	poetry run python src/train.py

.PHONY: cli-fit-model
cli-fit-model:
	poetry run python src/cli.py fit $(args)

.PHONY: cli-validate-model
cli-validate-model:
	poetry run python src/cli.py validate $(args)

.PHONY: cli-test-model
cli-test-model:
	poetry run python src/cli.py test $(args)

.PHONY: cli-predict-model
cli-predict-model:
	poetry run python src/cli.py predict $(args)

.PHONY: cli-help
cli-help:
	poetry run python src/cli.py --help

.PHONY: ui
ui:
	poetry run mlflow ui --backend-store-uri mlflow_runs 







