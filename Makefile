.PHONY: help setup data features train evaluate test lint format api clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install the package with all extras
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[viz,tune,api,dev]"

data:  ## Download the Home Credit dataset from Kaggle into data/raw
	$(PY) -m credit_risk.data.download

features:  ## Build the modelling table from raw tables
	$(PY) -m credit_risk.features.build

train:  ## Train the model with cross-validation
	$(PY) -m credit_risk.models.train

evaluate:  ## Evaluate the trained model and write reports
	$(PY) -m credit_risk.models.evaluate

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint and type-check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:  ## Auto-format the codebase
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

api:  ## Serve the scoring API locally
	.venv/bin/uvicorn credit_risk.api.main:app --reload --port 8000

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
