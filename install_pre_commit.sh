#!/bin/bash
# Pre-commit Hook Installation Script
# This script installs and configures pre-commit hooks for automated security scanning

set -e  # Exit on error

echo "==================================="
echo "Pre-commit Hook Installation"
echo "==================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "✓ Python found: $($PYTHON_CMD --version)"
echo ""

# Check if pip is installed
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ Error: pip is not installed"
    echo "Please install pip and try again"
    exit 1
fi

echo "✓ pip found: $($PYTHON_CMD -m pip --version)"
echo ""

# Install/upgrade pre-commit
echo "📦 Installing pre-commit..."
$PYTHON_CMD -m pip install --upgrade pre-commit

if [ $? -eq 0 ]; then
    echo "✓ pre-commit installed successfully"
else
    echo "❌ Failed to install pre-commit"
    exit 1
fi
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "✓ Git repository detected"
echo ""

# Install pre-commit hooks
echo "🔧 Installing git hooks..."
pre-commit install

if [ $? -eq 0 ]; then
    echo "✓ Git hooks installed successfully"
else
    echo "❌ Failed to install git hooks"
    exit 1
fi
echo ""

# Run pre-commit on all files as a test
echo "🧪 Running initial test on all files..."
echo "This may take a few minutes on first run..."
echo ""

pre-commit run --all-files

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✅ Setup Complete!"
    echo "==================================="
    echo ""
    echo "Pre-commit hooks are now active and will run automatically on:"
    echo "  • git commit (blocks commits if checks fail)"
    echo ""
    echo "Configured hooks:"
    echo "  • gitleaks - Scans for secrets and credentials"
    echo "  • pip-audit - Checks for vulnerable dependencies"
    echo ""
    echo "To run manually: pre-commit run --all-files"
    echo "To bypass hooks (emergency only): git commit --no-verify"
    echo ""
    echo "For more information, see PRE_COMMIT_SETUP.md"
else
    echo ""
    echo "==================================="
    echo "⚠️  Setup Complete with Warnings"
    echo "==================================="
    echo ""
    echo "Pre-commit hooks are installed, but some checks failed."
    echo "This is normal if there are existing issues in the codebase."
    echo ""
    echo "The hooks will still run on future commits."
    echo "Review the output above and fix any issues."
    echo ""
    echo "For more information, see PRE_COMMIT_SETUP.md"
fi

echo ""
