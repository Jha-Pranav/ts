# Variables
PYTHON := python3
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate
REQUIREMENTS := requirements.txt
SRC_DIR := src
TEST_DIR := tests

# Default target
.PHONY: all
all: install

# Create a virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)

# Install dependencies
.PHONY: install
install: venv
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r $(REQUIREMENTS)

# Run unit tests
.PHONY: test
test:
	$(ACTIVATE) && pytest $(TEST_DIR)

# Run linting (flake8)
.PHONY: lint
lint:
	$(ACTIVATE) && flake8 $(SRC_DIR) $(TEST_DIR)

# Run formatting (black)
.PHONY: format
format:
	$(ACTIVATE) && black $(SRC_DIR) $(TEST_DIR)

# Run static type checking (mypy)
.PHONY: type-check
type-check:
	$(ACTIVATE) && mypy $(SRC_DIR)

# Remove unnecessary files (cache, logs, etc.)
.PHONY: clean
clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -exec rm -rf {} +
	rm -rf $(VENV)

# Run the application (Modify as needed)
.PHONY: run
run:
	$(ACTIVATE) && python main.py

# Show available Make commands
.PHONY: help
help:
	@echo "Available make commands:"
	@echo "  make install     - Set up the virtual environment and install dependencies"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run code linting using flake8"
	@echo "  make format      - Format code using black"
	@echo "  make type-check  - Run static type checking with mypy"
	@echo "  make clean       - Remove cache, logs, and virtual environment"
	@echo "  make run         - Run the main application"
