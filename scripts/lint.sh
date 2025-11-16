#!/bin/bash

# Local development linting and formatting script
# Run this before committing code

set -e

echo "🔍 Running Ruff Linter..."
uv run ruff check . --fix

echo "🎨 Running Ruff Formatter..."
uv run ruff format .

echo "🔧 Running MyPy Type Checker..."
uv run mypy narration_generator.py --ignore-missing-imports

echo "✅ All checks passed! Code is ready to commit."