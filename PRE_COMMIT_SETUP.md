# Pre-Commit Security Hooks Setup Guide

This project uses [pre-commit](https://pre-commit.com/) to automatically run security scans before each commit, providing continuous protection against vulnerabilities and secrets.

## What Gets Scanned

The pre-commit hooks automatically run:

1. **Gitleaks** - Scans for hardcoded secrets, API keys, passwords, and credentials
2. **pip-audit** - Checks Python dependencies for known security vulnerabilities

## Quick Setup

### 1. Install Dependencies

First, ensure you have all project dependencies including pre-commit:

```bash
pip install -r requirements.txt
```

Or install pre-commit separately:

```bash
pip install pre-commit
```

### 2. Install Git Hooks

Install the pre-commit hooks into your local repository:

```bash
pre-commit install
```

This creates Git hooks in `.git/hooks/` that will run automatically before each commit.

### 3. Test the Setup

Run the hooks manually on all files to verify everything works:

```bash
pre-commit run --all-files
```

### 4. Verify with Test Commit

Test that hooks trigger on commit:

```bash
git commit --allow-empty -m "test: verify pre-commit hooks"
```

You should see both gitleaks and pip-audit execute before the commit proceeds.

## How It Works

When you run `git commit`, the pre-commit framework will:

1. **Block the commit** if any hooks fail
2. **Run gitleaks** to scan for secrets in your staged files
3. **Run pip-audit** to check requirements.txt for vulnerable packages
4. **Allow the commit** only if all checks pass

## Configuration

The hooks are configured in `.pre-commit-config.yaml`:

- **Gitleaks**: v8.18.0 - Detects secrets and credentials
- **pip-audit**: v2.6.1 - Scans Python dependencies for CVEs

## Troubleshooting

### Hooks Don't Run

If hooks don't execute on commit:

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Skip Hooks (Emergency Only)

To bypass hooks in an emergency (NOT recommended):

```bash
git commit --no-verify -m "your message"
```

⚠️ **Warning**: Only use `--no-verify` when absolutely necessary and ensure manual security review.

### Update Hooks

Update all hooks to their latest versions:

```bash
pre-commit autoupdate
```

### Clean Cache

If hooks behave unexpectedly, clear the cache:

```bash
pre-commit clean
```

## Manual Scans

You can also run security scans manually using the provided scripts:

**Unix/Linux/macOS:**
```bash
./run_security_scans.sh
```

**Windows:**
```cmd
run_security_scans.bat
```

## CI/CD Integration

The same `.pre-commit-config.yaml` can be used in CI/CD pipelines:

```yaml
# Example GitHub Actions
- uses: pre-commit/action@v3.0.0
```

## Benefits

✅ **Catch issues early** - Before code reaches the repository  
✅ **Consistent checks** - Same scans for all developers  
✅ **Fast feedback** - Know immediately if there's a problem  
✅ **Zero configuration** - Works automatically after setup  
✅ **Prevent accidents** - Stop secrets from being committed  

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)
- Project Security Policy: `SECURITY.md`

## Support

For issues or questions about security scanning:

1. Check `SECURITY_SCAN_RESULTS.md` for known issues
2. Review `SECURITY.md` for security policies
3. Run manual scans with `run_security_scans.sh` for detailed output
