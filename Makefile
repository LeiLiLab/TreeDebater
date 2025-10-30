.PHONY: clean deepclean install dev constraints black isort mypy ruff toml-sort lint pre-commit \
	auto-black auto-isort auto-toml-sort auto-lint \
	test-run test-run-offline test test-offline \
	build upload docs-autobuild changelog docs-gen docs-mypy docs-coverage docs

SHELL := /bin/bash

########################################################################################
# Variables
########################################################################################

# Use pipenv when not in CI and available
PIPRUN := $(shell [ "$$CI" != "true" ] && command -v pipenv > /dev/null 2>&1 && echo "pipenv run")

# Python version (major.minor)
PYTHON_VERSION := $(shell echo $${PYTHON_VERSION:-$$(python -V 2>&1 | cut -d ' ' -f 2)} | cut -d '.' -f 1,2)

# Constraints file by Python version
CONSTRAINTS_DIR := constraints
CONSTRAINTS_FILE := $(CONSTRAINTS_DIR)/$(PYTHON_VERSION).txt

# Public docs dir (compatible with ReadTheDocs)
PUBLIC_DIR := $(shell [ "$$READTHEDOCS" = "True" ] && echo "$$READTHEDOCS_OUTPUT/html" || echo "public")

# Changelog (optional)
CHANGELOG_URL := $(shell echo $${CI_PAGES_URL:-https://example.com/yourproject}/_sources/changelog.md.txt)
CHANGELOG_PATH := docs/changelog.md

########################################################################################
# Development Environment Management
########################################################################################

clean:
	-rm -rf \
		$(PUBLIC_DIR) \
		.coverage \
		.mypy_cache \
		.pytest_cache \
		.ruff_cache \
		Pipfile* \
		coverage.xml \
		dist \
		release-notes.md
	find . -name '*.egg-info' -print0 | xargs -0 rm -rf
	find . -name '*.pyc' -print0 | xargs -0 rm -f
	find . -name '*.swp' -print0 | xargs -0 rm -f
	find . -name '.DS_Store' -print0 | xargs -0 rm -f
	find . -name '__pycache__' -print0 | xargs -0 rm -rf

deepclean: clean
	if command -v pre-commit > /dev/null 2>&1; then pre-commit uninstall --hook-type pre-push; fi
	if command -v pipenv >/dev/null 2>&1 && pipenv --venv >/dev/null 2>&1; then pipenv --rm; fi

# Editable install (if this is a package). For this repo we default to requirements.txt
install:
	@if [ -f setup.py ] || [ -f pyproject.toml ]; then \
		$(PIPRUN) pip install -e . -c $(CONSTRAINTS_FILE) || $(PIPRUN) pip install -e . ; \
	else \
		$(PIPRUN) pip install -r requirements.txt ; \
	fi

# Developer setup: project deps + common dev tooling
dev:
	@if [ -f requirements.txt ]; then $(PIPRUN) pip install -r requirements.txt ; fi
	# Common dev tools (safe if already installed)
	$(PIPRUN) pip install -U \
		black isort ruff mypy \
		pytest coverage build twine \
		sphinx sphinx-autobuild git-changelog toml-sort
	@if [ "$(CI)" != "true" ] && command -v pre-commit > /dev/null 2>&1; then pre-commit install --hook-type pre-push; fi

# Generate constraints for current Python version
constraints: deepclean
	@mkdir -p $(CONSTRAINTS_DIR)
	@if [ -f setup.py ] || [ -f pyproject.toml ]; then \
		$(PIPRUN) --python $(PYTHON_VERSION) pip install --upgrade -e . ; \
	fi
	@if [ -f requirements.txt ]; then $(PIPRUN) pip install -r requirements.txt ; fi
	$(PIPRUN) pip freeze --exclude-editable > $(CONSTRAINTS_FILE)

########################################################################################
# Lint and pre-commit
########################################################################################

black:
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) python -m black --check --diff . -l 120

isort:
	@command -v isort >/dev/null 2>&1 || { echo "isort not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) python -m isort --check .

mypy:
	@command -v mypy >/dev/null 2>&1 || { echo "mypy not installed. Run 'make dev' first."; exit 1; }
	# Narrow the scope if needed
	$(PIPRUN) python -m mypy src || true

ruff:
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) ruff check src || true

toml-sort:
	@command -v toml-sort >/dev/null 2>&1 || { echo "toml-sort not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) toml-sort --check pyproject.toml || true

# Prioritize isort before black to avoid style conflicts
lint: mypy ruff isort black toml-sort

pre-commit:
	pre-commit run --all-files

########################################################################################
# Auto Lint
########################################################################################

auto-black:
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) python -m black . -l 120

auto-isort:
	@command -v isort >/dev/null 2>&1 || { echo "isort not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) python -m isort .

auto-toml-sort:
	@command -v toml-sort >/dev/null 2>&1 || { echo "toml-sort not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) toml-sort --in-place pyproject.toml >/dev/null 2>&1 || true

auto-lint: auto-isort auto-black auto-toml-sort

########################################################################################
# Test
########################################################################################

test-run:
	@command -v coverage >/dev/null 2>&1 || { echo "coverage not installed. Run 'make dev' first."; exit 1; }
	@if command -v pytest >/dev/null 2>&1; then \
		$(PIPRUN) python -m coverage erase; \
		$(PIPRUN) python -m coverage run --concurrency=multiprocessing -m pytest || true; \
		$(PIPRUN) python -m coverage combine; \
	else \
		echo "pytest not installed or no tests; skipping test-run."; \
	fi

test-run-offline:
	@command -v coverage >/dev/null 2>&1 || { echo "coverage not installed. Run 'make dev' first."; exit 1; }
	@if command -v pytest >/dev/null 2>&1; then \
		$(PIPRUN) python -m coverage erase; \
		$(PIPRUN) python -m coverage run --concurrency=multiprocessing -m pytest -m "offline" || true; \
		$(PIPRUN) python -m coverage combine; \
	else \
		echo "pytest not installed or no tests; skipping test-run-offline."; \
	fi

test: test-run
	$(PIPRUN) python -m coverage report --fail-under 20 || true
	$(PIPRUN) python -m coverage xml --fail-under 20 || true

test-offline: test-run-offline
	$(PIPRUN) python -m coverage report --fail-under 20 || true
	$(PIPRUN) python -m coverage xml --fail-under 20 || true

########################################################################################
# Package
########################################################################################

build:
	@command -v python >/dev/null 2>&1 || { echo "python not found"; exit 1; }
	$(PIPRUN) python -m build

upload:
	$(PIPRUN) python -m twine upload dist/*

########################################################################################
# Documentation (optional, only if docs/ exists)
########################################################################################

docs-autobuild:
	@if [ -d docs ]; then \
		$(PIPRUN) python -m sphinx_autobuild docs $(PUBLIC_DIR); \
	else \
		echo "No docs directory; skipping docs-autobuild."; \
	fi

changelog:
	@if wget -q --spider $(CHANGELOG_URL); then \
		echo "Existing Changelog found at '$(CHANGELOG_URL)', download for incremental generation."; \
		wget -q -O $(CHANGELOG_PATH) $(CHANGELOG_URL); \
	fi
	@command -v git-changelog >/dev/null 2>&1 || { echo "git-changelog not installed. Run 'make dev' first."; exit 1; }
	$(PIPRUN) LATEST_TAG=$$(git tag --sort=-creatordate | head -n 1); \
	git-changelog --bump $$LATEST_TAG -Tio docs/changelog.md -c conventional -s build,chore,ci,deps,doc,docs,feat,fix,perf,ref,refactor,revert,style,test,tests || true

release-notes:
	@command -v git-changelog >/dev/null 2>&1 || { echo "git-changelog not installed. Run 'make dev' first."; exit 1; }
	@$(PIPRUN) git-changelog --input $(CHANGELOG_PATH) --release-notes || true

docs-gen:
	@if [ -d docs ]; then \
		$(PIPRUN) python -m sphinx.cmd.build -W docs $(PUBLIC_DIR); \
	else \
		echo "No docs directory; skipping docs-gen."; \
	fi

docs-mypy: docs-gen
	@if [ -d docs ]; then \
		$(PIPRUN) python -m mypy src --html-report $(PUBLIC_DIR)/reports/mypy || true; \
	else \
		echo "No docs directory; skipping docs-mypy."; \
	fi

docs-coverage: test-run docs-gen
	@if [ -d docs ]; then \
		$(PIPRUN) python -m coverage html -d $(PUBLIC_DIR)/reports/coverage --fail-under 20 || true; \
	else \
		echo "No docs directory; skipping docs-coverage."; \
	fi

docs: changelog docs-gen docs-mypy docs-coverage

########################################################################################
# End
########################################################################################


