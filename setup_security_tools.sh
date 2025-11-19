#!/bin/bash
# Security Tools Installation Script
# This script installs pip-audit and gitleaks for security scanning

set -e  # Exit on error

echo "=========================================="
echo "Security Tools Setup"
echo "=========================================="
echo ""

# Install pip-audit
echo "Installing pip-audit..."
pip install pip-audit

# Verify pip-audit installation
if command -v pip-audit &> /dev/null; then
    echo "✓ pip-audit installed successfully"
    pip-audit --version
else
    echo "✗ pip-audit installation failed"
    exit 1
fi

echo ""

# Install gitleaks
echo "Installing gitleaks..."

# Detect OS and install accordingly
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS"
    # Check if wget or curl is available
    if command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -O"
    elif command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -L -o"
    else
        echo "✗ Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    # Download latest gitleaks binary for Linux
    GITLEAKS_VERSION="8.18.1"
    GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
    
    echo "Downloading gitleaks v${GITLEAKS_VERSION}..."
    $DOWNLOAD_CMD gitleaks.tar.gz "$GITLEAKS_URL"
    tar -xzf gitleaks.tar.gz
    chmod +x gitleaks
    
    # Move to /usr/local/bin if we have permission, otherwise suggest manual move
    if [ -w "/usr/local/bin" ]; then
        mv gitleaks /usr/local/bin/
        echo "✓ gitleaks installed to /usr/local/bin/"
    else
        echo "✓ gitleaks binary extracted to current directory"
        echo "  To install system-wide, run: sudo mv gitleaks /usr/local/bin/"
    fi
    
    rm -f gitleaks.tar.gz LICENSE

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    # Try to use Homebrew if available
    if command -v brew &> /dev/null; then
        echo "Using Homebrew to install gitleaks..."
        brew install gitleaks
    else
        echo "Homebrew not found. Downloading binary..."
        GITLEAKS_VERSION="8.18.1"
        GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_darwin_x64.tar.gz"
        
        if command -v curl &> /dev/null; then
            curl -L -o gitleaks.tar.gz "$GITLEAKS_URL"
            tar -xzf gitleaks.tar.gz
            chmod +x gitleaks
            
            if [ -w "/usr/local/bin" ]; then
                mv gitleaks /usr/local/bin/
                echo "✓ gitleaks installed to /usr/local/bin/"
            else
                echo "✓ gitleaks binary extracted to current directory"
                echo "  To install system-wide, run: sudo mv gitleaks /usr/local/bin/"
            fi
            
            rm -f gitleaks.tar.gz LICENSE
        else
            echo "✗ curl not found. Cannot download gitleaks."
            exit 1
        fi
    fi

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "Detected Windows (Git Bash/Cygwin)"
    echo "Please run setup_security_tools.bat instead on Windows"
    exit 1

else
    echo "Unsupported OS: $OSTYPE"
    echo "Please install gitleaks manually from: https://github.com/gitleaks/gitleaks/releases"
    exit 1
fi

echo ""

# Verify gitleaks installation
if command -v gitleaks &> /dev/null; then
    echo "✓ gitleaks installed successfully"
    gitleaks version
else
    echo "✗ gitleaks installation failed or not in PATH"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: pip-audit --version"
echo "2. Run: gitleaks version"
echo "3. See SECURITY.md for usage instructions"
