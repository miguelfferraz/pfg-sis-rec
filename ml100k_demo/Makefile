.PHONY: install install-dev run format check lint export-csv clean

PYTHON := python
PIP := pip
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate &&

install:
	$(ACTIVATE) $(PIP) install -e .

install-dev:
	$(ACTIVATE) $(PIP) install -e ".[dev]"

run:
	$(ACTIVATE) $(PYTHON) main.py

format:
	$(ACTIVATE) black .
	$(ACTIVATE) isort .

check:
	$(ACTIVATE) black --check .
	$(ACTIVATE) isort --check-only .

lint:
	$(ACTIVATE) flake8 .

export-csv:
	$(ACTIVATE) $(PYTHON) exploratory_analysis/main.py --export-csv recommendations.csv

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
