poetry/install:
	poetry install

poetry/setup:
ifdef POETRY_VERSION
	@echo "Found poetry version $(POETRY_VERSION)"
else
	@echo "Poetry not found, starting to install poetry"
	pip install pip -U
	pip install setuptools
	pip install poetry
	@echo "Installed poetry version" $(shell poetry --version)
endif

poetry/requirement-install: build/app
	pip install -r requirements.txt
	rm -f requirements.txt