PYTHON := python
PIP := pip
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate &&
PROJECT_PATH := "sis_rec_experiments"
DATASETS_PATH := "$(PROJECT_PATH)/datasets"


install:
	$(ACTIVATE) $(PIP) install -e .

install-dev:
	$(ACTIVATE) $(PIP) install -e ".[dev]"

run:
	$(ACTIVATE) $(PYTHON) $(PROJECT_PATH)/main.py

format:
	$(ACTIVATE) black .
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

datasets-extract: datasets-extract-amazon datasets-extract-anime datasets-extract-books datasets-extract-retail datasets-extract-steam

datasets-extract-amazon:
	@mkdir -p $(DATASETS_PATH)/extracted/AmazonMusic
	@if [ -f "$(DATASETS_PATH)/AmazonMusic.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/AmazonMusic.zip" -d "$(DATASETS_PATH)/extracted/AmazonMusic/"; \
	fi

datasets-extract-anime:
	@mkdir -p $(DATASETS_PATH)/extracted/anime
	@if [ -f "$(DATASETS_PATH)/Anime.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/Anime.zip" -d "$(DATASETS_PATH)/extracted/anime/"; \
	fi

datasets-extract-books:
	@mkdir -p $(DATASETS_PATH)/extracted/book_crossing
	@if [ -f "$(DATASETS_PATH)/BookCrossing.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/BookCrossing.zip" -d "$(DATASETS_PATH)/extracted/book_crossing/"; \
	fi

datasets-extract-retail:
	@mkdir -p $(DATASETS_PATH)/extracted/RetailRocket_Ecommerce
	@if [ -f "$(DATASETS_PATH)/RetailrocketEcommerce.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/RetailrocketEcommerce.zip" -d "$(DATASETS_PATH)/extracted/RetailRocket_Ecommerce/"; \
	fi

datasets-extract-steam:
	@mkdir -p $(DATASETS_PATH)/extracted/steam
	@if [ -f "$(DATASETS_PATH)/Steam.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/Steam.zip" -d "$(DATASETS_PATH)/extracted/steam/"; \
	fi

datasets-clean:
	@if [ -d "$(DATASETS_PATH)/extracted" ]; then \
		rm -rf $(DATASETS_PATH)/extracted; \
	fi

datasets-clean-all:
	@if [ -d "$(DATASETS_PATH)" ]; then \
		rm -rf $(DATASETS_PATH); \
	fi

.PHONY: install install-dev run format check lint clean datasets-extract datasets-extract-amazon datasets-extract-anime datasets-extract-books datasets-extract-retail datasets-extract-steam datasets-clean datasets-clean-all