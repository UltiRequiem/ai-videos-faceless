#!/bin/bash

# Narration Image Generator - Installation Script
# For macOS

set -e

echo "🎬 Narration Image Generator - Installation Script"
echo "=================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "🔧 Setting up virtual environment..."
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -e .

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Please create .env file with your API keys:"
    echo "   PEXELS_API_KEY=your_pexels_api_key_here"
    echo "   OPENAI_API_KEY=your_openai_api_key_here (optional)"
else
    echo "✅ .env file found"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Configure your .env file with API keys"
echo "3. Run the tool: python narration_generator.py examples/discipline_perception.txt"
echo ""
echo "For help: python narration_generator.py --help"