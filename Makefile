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

test:
	$(ACTIVATE) $(PYTHON) -m unittest discover -s $(PROJECT_PATH)/tests -p "test_*.py" -v

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

datasets-extract: datasets-extract-amazon datasets-extract-anime datasets-extract-books datasets-extract-retail datasets-extract-steam datasets-extract-movielens

datasets-extract-amazon:
	@mkdir -p $(DATASETS_PATH)/extracted
	@if [ -f "$(DATASETS_PATH)/AmazonMusic.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/AmazonMusic.zip" -d "$(DATASETS_PATH)/extracted/"; \
		mv "$(DATASETS_PATH)/extracted/AmazonMusic" "$(DATASETS_PATH)/extracted/AmazonMusic_temp" 2>/dev/null || true; \
		find "$(DATASETS_PATH)/extracted" -name "AmazonMusic*" -type d -exec mv {} "$(DATASETS_PATH)/extracted/AmazonMusic" \; 2>/dev/null || true; \
	fi

datasets-extract-anime:
	@mkdir -p $(DATASETS_PATH)/extracted
	@if [ -f "$(DATASETS_PATH)/Anime.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/Anime.zip" -d "$(DATASETS_PATH)/extracted/"; \
		find "$(DATASETS_PATH)/extracted" -name "*anime*" -type d -exec mv {} "$(DATASETS_PATH)/extracted/anime" \; 2>/dev/null || true; \
	fi

datasets-extract-books:
	@mkdir -p $(DATASETS_PATH)/extracted
	@if [ -f "$(DATASETS_PATH)/BookCrossing.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/BookCrossing.zip" -d "$(DATASETS_PATH)/extracted/"; \
		find "$(DATASETS_PATH)/extracted" -name "*book*" -type d -exec mv {} "$(DATASETS_PATH)/extracted/book_crossing" \; 2>/dev/null || true; \
	fi

datasets-extract-retail:
	@mkdir -p $(DATASETS_PATH)/extracted
	@if [ -f "$(DATASETS_PATH)/RetailrocketEcommerce.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/RetailrocketEcommerce.zip" -d "$(DATASETS_PATH)/extracted/"; \
		find "$(DATASETS_PATH)/extracted" -name "*Retail*" -type d -exec mv {} "$(DATASETS_PATH)/extracted/RetailRocket_Ecommerce" \; 2>/dev/null || true; \
	fi

datasets-extract-steam:
	@mkdir -p $(DATASETS_PATH)/extracted
	@if [ -f "$(DATASETS_PATH)/Steam.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/Steam.zip" -d "$(DATASETS_PATH)/extracted/"; \
		find "$(DATASETS_PATH)/extracted" -name "*steam*" -type d -exec mv {} "$(DATASETS_PATH)/extracted/steam" \; 2>/dev/null || true; \
	fi

datasets-extract-movielens:
	@mkdir -p $(DATASETS_PATH)/extracted/movielens
	@if [ -f "$(DATASETS_PATH)/MovieLens100k.zip" ]; then \
		unzip -q -o "$(DATASETS_PATH)/MovieLens100k.zip" -d "$(DATASETS_PATH)/extracted/movielens/"; \
	fi

datasets-clean:
	@if [ -d "$(DATASETS_PATH)/extracted" ]; then \
		rm -rf $(DATASETS_PATH)/extracted; \
	fi

datasets-clean-all:
	@if [ -d "$(DATASETS_PATH)" ]; then \
		rm -rf $(DATASETS_PATH); \
	fi

.PHONY: install install-dev run test format check lint clean datasets-extract datasets-extract-amazon datasets-extract-anime datasets-extract-books datasets-extract-retail datasets-extract-steam datasets-extract-movielens datasets-clean datasets-clean-all