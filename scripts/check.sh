#!/bin/bash

# Check code quality without making changes
# Use this in CI or to verify code quality

set -e

echo "🔍 Running Ruff Linter (check only)..."
uv run ruff check .

echo "🎨 Running Ruff Formatter (check only)..."
uv run ruff format --check .

echo "🔧 Running MyPy Type Checker..."
uv run mypy narration_generator.py --ignore-missing-imports

echo "📦 Testing imports..."
uv run python -c "import narration_generator; print('✅ Import successful')"

echo "✅ All checks passed!"