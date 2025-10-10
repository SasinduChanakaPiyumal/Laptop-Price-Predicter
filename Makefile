# Makefile for Python project with cross-platform support
# Provides standard developer targets for setup, formatting, linting, auditing, and cleanup

# Detect OS and set appropriate paths
ifeq ($(OS),Windows_NT)
	BIN := Scripts
	PYTHON_EXE := python.exe
	RM := powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	FIND_PYCACHE := powershell -Command "Get-ChildItem -Path . -Recurse -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	BIN := bin
	PYTHON_EXE := python
	RM := rm -rf
	FIND_PYCACHE := find . -name "__pycache__" -type d -prune -exec rm -rf {} +
endif

# Virtual environment configuration
VENV := .venv
VENV_BIN := $(VENV)/$(BIN)
PY := $(VENV_BIN)/$(PYTHON_EXE)
PIP := $(VENV_BIN)/pip

# Declare phony targets
.PHONY: help setup fmt lint nb-clean audit sec-scan clean

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      - Create venv, install dependencies, setup pre-commit and nbstripout"
	@echo "  fmt        - Format code with black and isort"
	@echo "  lint       - Run flake8 and bandit linters"
	@echo "  nb-clean   - Install nbstripout git filter for notebooks"
	@echo "  audit      - Run pip-audit on requirements.txt"
	@echo "  sec-scan   - Scan for secrets with gitleaks (optional)"
	@echo "  clean      - Remove Python caches and build artifacts"

# Setup target: create venv, install dependencies, setup tools
setup:
	@echo "==> Creating virtual environment with Python 3.11..."
	python3.11 -m venv $(VENV) || python -m venv $(VENV)
	@echo "==> Upgrading pip..."
	$(PY) -m pip install --upgrade pip
	@echo "==> Installing pip-tools..."
	$(PIP) install pip-tools
	@echo "==> Compiling requirements from requirements.in..."
	@if [ -f requirements.in ]; then \
		$(PY) -m piptools compile --generate-hashes --output-file=requirements.txt requirements.in; \
		echo "==> Syncing dependencies with pip-sync..."; \
		$(PY) -m piptools sync requirements.txt; \
	else \
		echo "Warning: requirements.in not found, skipping pip-tools compilation"; \
	fi
	@echo "==> Installing pre-commit hooks..."
	$(PIP) install pre-commit || true
	$(PY) -m pre_commit install || echo "Warning: pre-commit installation failed (continuing)"
	@echo "==> Installing nbstripout filter..."
	$(PIP) install nbstripout || true
	$(PY) -m nbstripout --install || echo "Warning: nbstripout installation failed (continuing)"
	@echo "==> Setup complete!"

# Format code with black and isort
fmt:
	@echo "==> Formatting code with black..."
	$(PY) -m black .
	@echo "==> Sorting imports with isort..."
	$(PY) -m isort .
	@echo "==> Formatting complete!"

# Lint code with flake8 and bandit
lint:
	@echo "==> Running flake8..."
	$(PY) -m flake8 .
	@echo "==> Running bandit security checks..."
	$(PY) -m bandit -r .
	@echo "==> Linting complete!"

# Install nbstripout filter (idempotent)
nb-clean:
	@echo "==> Installing nbstripout git filter..."
	$(PY) -m nbstripout --install
	@echo "==> nbstripout filter installed!"

# Run pip-audit on requirements.txt
audit:
	@echo "==> Running pip-audit on requirements.txt..."
	@if [ -f requirements.txt ]; then \
		$(PY) -m pip_audit -r requirements.txt; \
	else \
		echo "Warning: requirements.txt not found, skipping audit"; \
	fi
	@echo "==> Audit complete!"

# Scan for secrets with gitleaks (optional)
sec-scan:
	@echo "==> Checking for gitleaks..."
	@if command -v gitleaks >/dev/null 2>&1; then \
		echo "==> Running gitleaks..."; \
		gitleaks detect --source . --log-opts="--all" --report-format sarif --report-path gitleaks.sarif; \
		echo "==> Security scan complete! Report: gitleaks.sarif"; \
	else \
		echo "==> gitleaks not found in PATH"; \
		echo "==> To install gitleaks, download from:"; \
		echo "    https://github.com/gitleaks/gitleaks/releases"; \
		echo "==> Skipping secret scanning (optional)"; \
	fi
	@exit 0

# Clean Python caches and build artifacts
clean:
	@echo "==> Cleaning Python caches and build artifacts..."
	$(FIND_PYCACHE)
	$(RM) .pytest_cache
	$(RM) .mypy_cache
	$(RM) build
	$(RM) dist
	$(RM) *.egg-info
	@echo "==> Cleanup complete!"
