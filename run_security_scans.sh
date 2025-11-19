#!/bin/bash
# Security Scans Execution Script
# This script runs pip-audit and gitleaks scans to establish baseline security posture
# Generates both JSON (human-readable) and SARIF (tool-integration) formats

set -e  # Exit on error

echo "=========================================="
echo "Security Baseline Scans"
echo "=========================================="
echo ""

# Verify tools are installed
echo "Verifying security tools installation..."

if ! command -v pip-audit &> /dev/null; then
    echo "✗ pip-audit not found. Please run setup_security_tools.sh first."
    exit 1
fi

if ! command -v gitleaks &> /dev/null; then
    echo "✗ gitleaks not found. Please run setup_security_tools.sh first."
    exit 1
fi

echo "✓ pip-audit found: $(pip-audit --version)"
echo "✓ gitleaks found: $(gitleaks version | head -n1)"
echo ""

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "✗ requirements.txt not found. Cannot run pip-audit."
    exit 1
fi

echo "=========================================="
echo "1. Running pip-audit (JSON format)..."
echo "=========================================="
pip-audit --desc --format json --output pip-audit-results.json
echo "✓ Generated: pip-audit-results.json"
echo ""

echo "=========================================="
echo "2. Running pip-audit (SARIF format)..."
echo "=========================================="
pip-audit --format sarif --output pip-audit.sarif
echo "✓ Generated: pip-audit.sarif"
echo ""

echo "=========================================="
echo "3. Running gitleaks (JSON format)..."
echo "=========================================="
gitleaks detect --no-git --verbose --report-path gitleaks-report.json || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        echo "⚠ gitleaks detected potential secrets (exit code 1)"
        echo "✓ Generated: gitleaks-report.json"
    else
        echo "✗ gitleaks failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
}
if [ $? -eq 0 ]; then
    echo "✓ Generated: gitleaks-report.json (no secrets detected)"
fi
echo ""

echo "=========================================="
echo "4. Running gitleaks (SARIF format)..."
echo "=========================================="
gitleaks detect --no-git --report-format sarif --report-path gitleaks.sarif || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        echo "⚠ gitleaks detected potential secrets (exit code 1)"
        echo "✓ Generated: gitleaks.sarif"
    else
        echo "✗ gitleaks failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
}
if [ $? -eq 0 ]; then
    echo "✓ Generated: gitleaks.sarif (no secrets detected)"
fi
echo ""

echo "=========================================="
echo "Scan Summary"
echo "=========================================="
echo "Output files generated:"
echo "  - pip-audit-results.json (vulnerability report)"
echo "  - pip-audit.sarif (SARIF format for tools)"
echo "  - gitleaks-report.json (secrets detection report)"
echo "  - gitleaks.sarif (SARIF format for tools)"
echo ""
echo "Next steps:"
echo "1. Review pip-audit-results.json for vulnerabilities"
echo "2. Review gitleaks-report.json for detected secrets"
echo "3. Document findings in SECURITY_SCAN_RESULTS.md"
echo "4. Address any CRITICAL or HIGH severity issues"
echo ""
echo "Note: These files are git-ignored and will not be committed."
echo "=========================================="
