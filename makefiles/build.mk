build/setup: poetry/setup poetry/install

build/app:
	pip install poetry-plugin-export
	poetry export --without-hashes -f requirements.txt > requirements.txt