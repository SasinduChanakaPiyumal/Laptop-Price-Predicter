# Project Setup Instructions

## Dependency Management

This project uses `pip-tools` for reproducible dependency management with hash verification for security.

### Prerequisites

Ensure you have the required tools installed:

```bash
pip install pip-tools pip-audit
```

### Compiling Dependencies

After modifying `requirements.in`, regenerate `requirements.txt` with hashes:

```bash
pip-compile requirements.in --generate-hashes
```

This will:
- Resolve all dependency versions
- Generate SHA256 hashes for each package
- Create a pinned `requirements.txt` file

### Security Scanning

Run vulnerability scanning on the compiled requirements:

```bash
pip-audit -r requirements.txt
```

This checks all dependencies against known CVE databases and reports any security vulnerabilities.

### Installing Dependencies

Once `requirements.txt` is generated and audited:

```bash
pip install -r requirements.txt
```

The hash verification ensures you're installing exactly the packages that were audited.

## Directory Structure

The project is organized as follows:

```
.
├── scrapers/          # Scraper modules (base classes and site-specific implementations)
├── data/              # Scraping outputs (CSV files, SQLite database) - gitignored
├── config/            # YAML configuration files
├── logs/              # Rotating log files - gitignored
├── tests/             # Unit and integration tests
├── tools/             # Utility scripts and tools
└── requirements.in    # Dependency specifications
```

## Next Steps

1. Run `pip-compile requirements.in --generate-hashes` to update requirements.txt
2. Run `pip-audit -r requirements.txt` to verify no vulnerabilities exist
3. Address any vulnerabilities if found
4. Install dependencies with `pip install -r requirements.txt`
