#!/bin/bash

# Dog vs Cat Classification - Setup Script
# This script creates a virtual environment and installs all required dependencies

echo "Dog vs Cat Classification Setup"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt


echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "   source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download dataset: kaggle competitions download -c dogs-vs-cats"
echo "2. Split data: python split_data.py"
echo "3. Train model: python train_model.py" 