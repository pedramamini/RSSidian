#!/bin/bash
# Setup development environment for RSSidian

set -e

# Create and activate virtual environment
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install hatch
pip install -e .

echo "Development environment setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Run with: rssidian [command]"