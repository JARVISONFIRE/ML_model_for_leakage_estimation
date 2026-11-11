#!/usr/bin/env bash
# Environment setup script for ML-Driven PVT Leakage Estimation Project
set -e

echo "=== Step 1: Checking NGSPICE installation ==="
if ! command -v ngspice &> /dev/null; then
    echo "[!] NGSPICE not found."
    echo "    On Arch Linux, please install via: sudo pacman -S ngspice"
    echo "    On Ubuntu/Debian, install via:    sudo apt update && sudo apt install -y ngspice"
    exit 1
else
    echo "[+] NGSPICE is installed: $(which ngspice)"
fi

echo ""
echo "=== Step 2: Creating Python Virtual Environment ==="
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[+] Created virtualenv at $VENV_DIR"
else
    echo "[+] Virtualenv already exists at $VENV_DIR"
fi

echo ""
echo "=== Step 3: Installing Python Packages ==="
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup Complete! ==="
echo "Activate environment using: source .venv/bin/activate"
