POETRY_VERSION :=$(shell poetry --version)

include makefiles/build.mk
include makefiles/poetry.mk
include makefiles/quality.mk

build: build/setup build/app

test: build/setup tests/coverage

format: format/code-style check/code-style