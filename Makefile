.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          		    Show this help"
	@echo "  torch-info   			Show torch information related to the project"
	@echo "  clean-code       		Clean the project"
	@echo "  test-code          	Run the tests"
	@echo "  coverage      			Run the tests with coverage"
	@echo "  lint-code          	Run the linter"
	@echo "  format-code        	Run the formatter"
	@echo "  train           	Fit model with as specified in src/train.py"
	@echo "  cli-fit-model          Fit the model using PyTorch Lightning CLI"
	@echo "  cli-validate-model     Perform one evaluation epoch over the validation set using PyTorch Lightning CLI"
	@echo "  cli-test-model         Perform one evaluation epoch over the test set using PyTorch Lightning CLI"
	@echo "  cli-predict-model      Run inference on your data. This will call the model forward function to compute predictions using PyTorch Lightning CLI."
	@echo "  lightning-help     	Show the help menu of the Lightning CLI"
	@echo "  ui   					Open the MLFlow UI"


.PHONY: torch-info
torch-info:
	poetry run python src/utility/log_pytorch_info.py

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
	poetry run flake8 src tests

.PHONY: format-code
format-code:
	poetry run black src tests

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







