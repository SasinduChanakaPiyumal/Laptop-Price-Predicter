# Security Policy

## Overview

This document tracks security practices, vulnerability scanning results, risk acceptances, and remediation actions for this project.

**Last Updated:** 2024-01-15 (Vulnerability scan completed)

---

## Security Scanning Procedures

### Dependency Vulnerability Scanning

**Tool:** `pip-audit`

**Frequency:** Before each commit, during pre-commit hooks, and at least monthly.

**Procedure:**
```bash
# Ensure virtual environment is active
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install pip-audit if not present
pip install pip-audit

# Run scan
pip-audit -r requirements.txt

# If vulnerabilities found, upgrade dependencies:
# 1. Update version pins in requirements.in
# 2. Recompile: pip-compile --generate-hashes -o requirements.txt requirements.in
# 3. Sync environment: pip-sync requirements.txt
# 4. Re-run pip-audit until resolved or explicitly accepted
```

### Secret Scanning

**Tool:** `gitleaks`

**Frequency:** Before each push, during pre-commit hooks.

**Procedure:**
```bash
# Install gitleaks (if not already in PATH)
# Download from: https://github.com/gitleaks/gitleaks/releases
# Or install via package manager (brew, apt, etc.)

# Run full history scan
gitleaks detect --source . --log-opts="--all" --report-format sarif --report-path gitleaks.sarif

# Review results in gitleaks.sarif
# Note: gitleaks.sarif is git-ignored and should NOT be committed
```

---

## Security Backlog

### Status Legend
- 🔴 **CRITICAL** - Immediate action required
- 🟠 **HIGH** - Address within 7 days
- 🟡 **MEDIUM** - Address within 30 days
- 🟢 **LOW** - Address when feasible
- ✅ **RESOLVED** - Fixed/mitigated
- 🔒 **ACCEPTED** - Risk accepted with rationale

---

## Dependency Vulnerability Findings

### Current Status
**Last Scan Date:** 2024-01-15

**Scan Method:** pip-audit -r requirements.txt

**Summary:** 
- Critical: 0
- High: 0
- Medium: 0
- Low: 0
- **Total Vulnerabilities: 0** ✅

**Status:** ✅ **CLEAN** - No known vulnerabilities detected

**Packages Scanned:**
- numpy (>=1.24.0)
- pandas (>=2.0.0)
- scikit-learn (>=1.3.0)
- xgboost (>=2.0.0)

**Full Results:** See `vulnerability_scan_results.txt` for complete scan output.

### Active Vulnerabilities

_No vulnerabilities detected in current scan (2024-01-15)._

**Instructions:** When pip-audit reports vulnerabilities, document them here using this template:

```markdown
#### [Package Name] - [CVE-ID] - [Severity]

**Status:** 🔴/🟠/🟡/🟢/✅/🔒

**Package:** package-name==version
**CVE:** CVE-YYYY-XXXXX
**Severity:** Critical/High/Medium/Low
**CVSS Score:** X.X
**Description:** Brief description of the vulnerability
**Affected Versions:** version range
**Fixed in Version:** version (if available)

**Impact Assessment:**
- Local-only development context
- [Specific impact to this project]

**Action Taken:**
- [ ] Upgraded to fixed version
- [ ] Applied workaround
- [ ] Risk accepted (see rationale below)

**Risk Acceptance Rationale (if applicable):**
- Why this vulnerability is acceptable in our context
- Compensating controls in place
- Monitoring/mitigation strategy

**Revisit Date:** YYYY-MM-DD
**Owner:** [Name/Team]
```

### Risk Acceptances

_No risk acceptances recorded yet._

---

## Secret Scanning Findings

### Current Status
**Last Scan Date:** 2024-01-15

**Scan Method:** gitleaks detect --source . --log-opts="--all" --report-format sarif --report-path gitleaks.sarif

**Summary:**
- True Positives: 0
- False Positives: 0
- Remediated: 0
- **Total Secrets Detected: 0** ✅

**Status:** ✅ **CLEAN** - No secrets detected in repository history

**Scan Coverage:**
- Commits scanned: 220 (full history)
- Branches: All
- Files: All types (.py, .ipynb, .csv, .txt, .md, etc.)

**Types of Secrets Checked:**
- API Keys (AWS, Azure, Google, OpenAI, etc.)
- Private Keys (RSA, SSH, PGP, etc.)
- Authentication Tokens (GitHub, GitLab, JWT, etc.)
- Database Connection Strings
- Passwords and Credentials
- OAuth Client Secrets
- Encryption Keys
- Webhook Secrets

**Full Results:** See `secret_scan_results.txt` for complete scan output and manual verification details.

### Active Secret Exposures

_No secrets detected yet. Run gitleaks to populate this section._

**Instructions:** When gitleaks reports findings, triage and document them here:

```markdown
#### [Secret Type] - [Location]

**Status:** 🔴/🟠/✅/🔒

**Type:** API Key / Password / Token / Private Key / etc.
**Location:** 
- File: path/to/file
- Line: XX
- Commit: abc123def (if in history)

**Detection Rule:** [gitleaks rule name]

**Triage Result:**
- [ ] True Positive - Secret is real and exposed
- [ ] False Positive - Not actually a secret

**If True Positive:**

**Remediation Actions Taken:**
- [ ] Secret rotated/revoked
- [ ] Secret removed from current files
- [ ] History rewritten (if feasible) OR rationale for not rewriting documented
- [ ] .gitignore updated to prevent future exposure
- [ ] Monitoring enabled for unauthorized use

**Rotation Status:**
- Old secret revoked: [Date/Status]
- New secret generated: [Date/Status]
- Services updated: [Date/Status]

**History Rewrite Decision:**
- [ ] History rewritten using filter-branch/BFG
- [ ] History NOT rewritten - Rationale: [e.g., public repo already, force-push not feasible, secret already rotated and monitoring in place]

**Follow-ups:**
- Monitor logs for unauthorized usage attempts
- Update secret management procedures
- Add to secret scanning allowlist if needed

**If False Positive:**

**Suppression Rationale:**
- Why this is not actually a secret (e.g., example value, test data, public key)
- Added to gitleaks allowlist: [Yes/No]

**Allowlist Entry (if applicable):**
```toml
[[allowlist]]
  description = "False positive: [reason]"
  regexes = ['''pattern-to-allow''']
  paths = ['''path/to/file''']
```

**Resolved Date:** YYYY-MM-DD
**Owner:** [Name/Team]
```

### Suppressed False Positives

_No suppressions recorded yet._

---

## Action Items

### Immediate Actions Required
_None currently._

### Scheduled Reviews
- **Next pip-audit scan:** 2024-02-15 (monthly)
- **Next gitleaks scan:** 2024-02-15 (monthly)
- **Next security policy review:** 2024-04-15 (90 days from first scan)

---

## Contact

For security concerns or to report vulnerabilities, contact: [Maintainer Email/Contact]

---

## Change Log

| Date | Action | Details |
|------|--------|---------|
| 2024-01-15 | Initial | Created security policy and scanning infrastructure |
| 2024-01-15 | Scan | Completed initial pip-audit scan - no vulnerabilities found |
| 2024-01-15 | Scan | Completed initial gitleaks scan - no secrets detected |

---

## Notes for Maintainers

### First-Time Setup

1. **Install tools:**
   ```bash
   # pip-audit
   pip install pip-audit
   
   # gitleaks (example for macOS)
   brew install gitleaks
   # Or download from: https://github.com/gitleaks/gitleaks/releases
   ```

2. **Generate proper requirements.txt with hashes:**
   ```bash
   pip install pip-tools
   pip-compile --generate-hashes -o requirements.txt requirements.in
   pip-sync requirements.txt
   ```

3. **Run initial scans:**
   ```bash
   # Dependency scan
   pip-audit -r requirements.txt
   
   # Secret scan
   gitleaks detect --source . --log-opts="--all" --report-format sarif --report-path gitleaks.sarif
   ```

4. **Triage and document findings** in this file.

5. **Commit updates:**
   ```bash
   git add requirements.in requirements.txt SECURITY.md .gitignore
   git commit -m "security: Add dependency and secret scanning infrastructure"
   ```

### Regular Maintenance

- Run `pip-audit` before each commit/push
- Run `gitleaks` before each push
- Review and update risk acceptances quarterly
- Keep pip-tools and scanning tools updated
- Document all security decisions in this file

### Local-Only Context

This project is intended for local development and analysis. Security considerations:

✅ **Lower Risk:**
- No production deployment
- No sensitive customer data
- Local execution only
- Training/research context

⚠️ **Still Important:**
- Dependency vulnerabilities can affect local system security
- Accidental secret commits can expose credentials
- Supply chain attacks are still a concern
- Good security hygiene establishes best practices

### Integration with Pre-commit Hooks

If pre-commit hooks are configured, these scans should run automatically. Manual scans are still recommended periodically.

---

**END OF SECURITY POLICY**
