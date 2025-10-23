#!/bin/bash
set -e

echo "=========================================="
echo "Setting up SLURM Python environment"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Create virtual environment using uv
echo ""
echo "Creating virtual environment with uv..."
uv venv .venv

# Activate the virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from pyproject.toml
echo ""
echo "Installing dependencies from pyproject.toml..."
uv pip install -e ..

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Virtual environment location: $SCRIPT_DIR/.venv"
echo ""
echo "To activate this environment manually:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
echo ""
echo "You can now submit SLURM jobs using:"
echo "  python submit_slurm_job.py ../slurm/<your_script>.sbatch"
echo ""

