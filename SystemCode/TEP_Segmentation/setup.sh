#!/bin/bash
# ===============================
# TEP Segmentation - Development Environment Setup (macOS/Linux)
# ===============================

# Check for Python 3.12
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION != *"3.12"* ]]; then
    echo "Please install Python 3.12 before running this script."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements-dev.txt

echo ""
echo "==============================="
echo "Setup complete!"
echo "==============================="