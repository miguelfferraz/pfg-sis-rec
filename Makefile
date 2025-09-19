PYTHON := python
PIP := pip
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate &&
PROJECT_PATH := "sis_rec_experiments"

install:
	$(ACTIVATE) $(PIP) install -e .

install-dev:
	$(ACTIVATE) $(PIP) install -e ".[dev]"

run:
	$(ACTIVATE) $(PYTHON) $(PROJECT_PATH)/main.py

format:
	$(ACTIVATE) black $
	$(ACTIVATE) isort .

check:
	$(ACTIVATE) black --check .
	$(ACTIVATE) isort --check-only .

lint:
	$(ACTIVATE) flake8 .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
		find . -type d -name "*.egg-info" -exec rm -rf {} +