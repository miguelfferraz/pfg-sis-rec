tests/coverage:
	poetry run pytest tests/ --cov=src --cov-report xml --cov-report term --cov-report html --no-cov-on-fail --cov-fail-under=90

check/code-style:
	poetry run unimport --check src tests
	poetry run isort --check src tests
	poetry run black --check src tests

format/code-style:
	poetry run unimport -r src tests
	poetry run isort src tests
	poetry run black src tests