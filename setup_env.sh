#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the virtual environment directory
VENV_DIR=".venv"

echo "Checking for existing virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
    echo "To recreate it, remove the '$VENV_DIR' directory and rerun this script."
else
    echo "Creating virtual environment '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

echo "Activating virtual environment..."
source "$VENV_DIR"/bin/activate
echo "Virtual environment activated."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed."

echo "Setup complete. To deactivate the environment, run 'deactivate'."
echo "To activate it again later, run 'source $VENV_DIR/bin/activate'."
