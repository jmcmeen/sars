# Makefile for sars development tasks.
# Dependency resolution and environment management use uv (https://docs.astral.sh/uv/).

.DEFAULT_GOAL := help

UV ?= uv
SRC := src
TESTS := tests

.PHONY: help install dev sync lock upgrade shell \
        test test-fast cov \
        lint fmt fmt-check check \
        clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## --- Environment & dependencies (uv) ---

install: ## Create venv and install the project with dev tooling
	$(UV) sync --extra dev

dev: install ## Alias for install

sync: ## Sync the environment to the lockfile (incl. dev tooling)
	$(UV) sync --extra dev

lock: ## Resolve and write uv.lock
	$(UV) lock

upgrade: ## Upgrade locked dependencies to latest allowed versions
	$(UV) lock --upgrade

shell: ## Spawn a subshell with the project venv activated
	$(UV) run $${SHELL:-bash}

## --- Testing ---

test: ## Run the full test suite
	$(UV) run --extra dev pytest

test-fast: ## Run tests, skipping slow/integration markers
	$(UV) run --extra dev pytest -m "not slow and not integration"

cov: ## Run tests with coverage report
	$(UV) run --extra dev pytest --cov --cov-report=term-missing

## --- Linting & formatting (ruff) ---

lint: ## Lint with ruff
	$(UV) run --extra dev ruff check $(SRC) $(TESTS)

fmt: ## Auto-format and apply lint fixes
	$(UV) run --extra dev ruff format $(SRC) $(TESTS)
	$(UV) run --extra dev ruff check --fix $(SRC) $(TESTS)

fmt-check: ## Check formatting without modifying files
	$(UV) run --extra dev ruff format --check $(SRC) $(TESTS)

check: lint fmt-check test ## Run lint, format check, and tests

## --- Housekeeping ---

clean: ## Remove caches and test/coverage build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist
	find . -name .venv -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
