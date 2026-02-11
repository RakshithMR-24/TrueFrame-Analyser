#!/bin/bash

echo "========================================="
echo "  Setting up TrueFrame Analyser Project  "
echo "========================================="

# Step 1: Check Python
echo "[1/5] Checking Python installation..."
python3 --version || { echo "Python3 is not installed. Please install Python 3.9+ first."; exit 1; }

# Step 2: Create Virtual Environment
echo "[2/5] Creating virtual environment..."
python3 -m venv venv

# Step 3: Activate Virtual Environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Step 4: Upgrade pip
echo "[4/5] Upgrading pip..."
pip install --upgrade pip

# Step 5: Install Requirements
echo "[5/5] Installing required packages..."
pip install -r requirements.txt

echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "To run the project:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "TrueFrame Analyser is ready!"
