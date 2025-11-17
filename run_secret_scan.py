#!/usr/bin/env python3
"""
Secret Scanning Script

This script automates the secret scanning process as defined in SECURITY.md
using gitleaks to scan the full Git repository history.

Usage:
    python run_secret_scan.py

Output:
    - gitleaks.sarif: SARIF format scan results (git-ignored)
    - secret_scan_results.txt: Human-readable summary
    - Console: Summary of findings

Requirements:
    - gitleaks must be installed (https://github.com/gitleaks/gitleaks)
    - Git repository must be initialized
"""

import subprocess
import sys
import json
from datetime import datetime
import os
from pathlib import Path


def check_gitleaks_installed():
    """Check if gitleaks is installed."""
    try:
        result = subprocess.run(
            ['gitleaks', 'version'],
            capture_output=True,
            check=True,
            text=True
        )
        version = result.stdout.strip()
        print(f"✓ gitleaks is installed: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ gitleaks not found")
        print("\nTo install gitleaks:")
        print("  macOS:   brew install gitleaks")
        print("  Linux:   Download from https://github.com/gitleaks/gitleaks/releases")
        print("  Windows: Download from https://github.com/gitleaks/gitleaks/releases")
        print("\nOr visit: https://github.com/gitleaks/gitleaks#installing")
        return False


def check_git_repository():
    """Check if current directory is a Git repository."""
    if not Path('.git').exists():
        print("✗ Error: Not a Git repository")
        print("  Initialize with: git init")
        return False
    print("✓ Git repository detected")
    return True


def run_secret_scan():
    """Execute gitleaks scan against full repository history."""
    print("\n" + "="*60)
    print("SECRET SCANNING - FULL REPOSITORY HISTORY")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tool: gitleaks")
    print(f"Scope: Full Git history (--log-opts='--all')")
    print("="*60 + "\n")
    
    try:
        # Run gitleaks detect with full history scan
        print("Running gitleaks scan (this may take a moment)...")
        result = subprocess.run(
            [
                'gitleaks', 'detect',
                '--source', '.',
                '--log-opts=--all',
                '--report-format', 'sarif',
                '--report-path', 'gitleaks.sarif',
                '--verbose'
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Determine scan status
        # gitleaks exit codes:
        # 0 = no leaks found
        # 1 = leaks found
        # other = error
        
        if result.returncode == 0:
            status = "CLEAN - No secrets detected"
            status_emoji = "✓"
        elif result.returncode == 1:
            status = "SECRETS DETECTED"
            status_emoji = "⚠"
        else:
            status = f"ERROR (exit code: {result.returncode})"
            status_emoji = "✗"
        
        # Parse SARIF output if it exists
        findings_summary = parse_sarif_results()
        
        # Create comprehensive output
        output = f"""Secret Scan Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Command: gitleaks detect --source . --log-opts="--all" --report-format sarif --report-path gitleaks.sarif

{'='*60}
STATUS: {status}
{'='*60}

{'='*60}
STDOUT:
{'='*60}
{result.stdout}

{'='*60}
STDERR:
{'='*60}
{result.stderr}

{'='*60}
EXIT CODE: {result.returncode}
{'='*60}

{findings_summary}
"""
        
        # Save results to file
        with open('secret_scan_results.txt', 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"\n{status_emoji} Scan completed: {status}")
        print(f"✓ Results saved to: secret_scan_results.txt")
        
        if result.returncode == 1:
            print(f"✓ SARIF report saved to: gitleaks.sarif (git-ignored)")
        
        # Print summary
        print("\n" + "="*60)
        print("SCAN OUTPUT:")
        print("="*60)
        print(result.stdout if result.stdout else "(no output)")
        
        if result.stderr:
            print("\n" + "="*60)
            print("ERRORS/WARNINGS:")
            print("="*60)
            print(result.stderr)
        
        # Print findings summary
        if findings_summary:
            print("\n" + findings_summary)
        
        # Status-specific messages
        if result.returncode == 0:
            print("\n✓ SUCCESS: No secrets found in repository history!")
        elif result.returncode == 1:
            print("\n⚠ SECRETS DETECTED")
            print("→ Review gitleaks.sarif and secret_scan_results.txt for details")
            print("→ Triage each finding (true positive vs false positive)")
            print("→ Update SECURITY.md with findings")
            print("\nFor each TRUE POSITIVE secret:")
            print("  1. Rotate/revoke the exposed secret immediately")
            print("  2. Remove from current files")
            print("  3. Consider rewriting history (git filter-branch/BFG)")
            print("  4. Update .gitignore to prevent future exposure")
        else:
            print(f"\n✗ ERROR: Scan failed with exit code {result.returncode}")
            print("→ Check stderr output above for details")
        
        return output
        
    except subprocess.TimeoutExpired:
        error_msg = "✗ Error: Scan timed out after 10 minutes"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"✗ Error running scan: {e}"
        print(error_msg)
        return error_msg


def parse_sarif_results():
    """Parse SARIF output file and return formatted summary."""
    sarif_path = Path('gitleaks.sarif')
    
    if not sarif_path.exists():
        return "\n(No SARIF file generated - no findings)"
    
    try:
        with open(sarif_path, 'r', encoding='utf-8') as f:
            sarif_data = json.load(f)
        
        # Extract findings from SARIF format
        runs = sarif_data.get('runs', [])
        if not runs:
            return "\n(SARIF file contains no runs)"
        
        results = runs[0].get('results', [])
        
        if not results:
            return "\n(SARIF file contains no results)"
        
        summary = "\n" + "="*60
        summary += "\nFINDINGS SUMMARY:"
        summary += "\n" + "="*60
        summary += f"\nTotal Findings: {len(results)}\n"
        
        # Group findings by type
        findings_by_type = {}
        for result in results:
            rule_id = result.get('ruleId', 'unknown')
            if rule_id not in findings_by_type:
                findings_by_type[rule_id] = []
            findings_by_type[rule_id].append(result)
        
        summary += f"\nFindings by Type:"
        for rule_id, findings in findings_by_type.items():
            summary += f"\n  - {rule_id}: {len(findings)} finding(s)"
        
        summary += "\n\nDetailed Findings:\n"
        summary += "-" * 60 + "\n"
        
        for idx, result in enumerate(results, 1):
            rule_id = result.get('ruleId', 'unknown')
            message = result.get('message', {}).get('text', 'No description')
            locations = result.get('locations', [])
            
            summary += f"\n{idx}. {rule_id}"
            summary += f"\n   Message: {message}"
            
            for loc in locations:
                physical_loc = loc.get('physicalLocation', {})
                artifact = physical_loc.get('artifactLocation', {})
                region = physical_loc.get('region', {})
                
                file_path = artifact.get('uri', 'unknown file')
                start_line = region.get('startLine', '?')
                
                summary += f"\n   Location: {file_path}:{start_line}"
            
            summary += "\n   " + "-" * 58
        
        summary += "\n\n⚠ IMPORTANT: Review each finding to determine if it is:"
        summary += "\n  - TRUE POSITIVE: A real secret that needs rotation"
        summary += "\n  - FALSE POSITIVE: Not actually a secret (test data, example, etc.)"
        
        return summary
        
    except json.JSONDecodeError as e:
        return f"\n✗ Error parsing SARIF file: {e}"
    except Exception as e:
        return f"\n✗ Error reading SARIF file: {e}"


def main():
    """Main execution function."""
    print("\nStarting Secret Scan Process...")
    print("Following procedures defined in SECURITY.md\n")
    
    # Step 1: Check prerequisites
    if not check_git_repository():
        print("\n✗ Cannot proceed without Git repository")
        sys.exit(1)
    
    if not check_gitleaks_installed():
        print("\n✗ Cannot proceed without gitleaks")
        sys.exit(1)
    
    # Step 2: Run scan
    result = run_secret_scan()
    
    if result is None:
        sys.exit(1)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Review secret_scan_results.txt")
    print("2. If findings exist, review gitleaks.sarif for details")
    print("3. Triage each finding (true vs false positive)")
    print("4. Update SECURITY.md with findings and actions taken")
    print("5. For true positives:")
    print("   a. Rotate/revoke the secret immediately")
    print("   b. Remove from current codebase")
    print("   c. Consider history rewriting (if feasible)")
    print("   d. Add monitoring for unauthorized use")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
