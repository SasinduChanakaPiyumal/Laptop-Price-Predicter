# Security Policy

## Overview

This document tracks security practices, vulnerability scanning results, risk acceptances, and remediation actions for this project.

**Last Updated:** 2024-01-XX (Update date when scans are run)

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
**Last Scan Date:** [PENDING - Run `pip-audit -r requirements.txt`]

**Summary:** 
- Critical: 0
- High: 0
- Medium: 0
- Low: 0

### Active Vulnerabilities

_No vulnerabilities recorded yet. Run pip-audit to populate this section._

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
**Last Scan Date:** [PENDING - Run `gitleaks detect`]

**Summary:**
- True Positives: 0
- False Positives: 0
- Remediated: 0

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
- **Next pip-audit scan:** [Schedule after first scan]
- **Next gitleaks scan:** [Schedule after first scan]
- **Next security policy review:** [90 days from first scan]

---

## Contact

For security concerns or to report vulnerabilities, contact: [Maintainer Email/Contact]

---

## Change Log

| Date | Action | Details |
|------|--------|---------|
| 2024-01-XX | Initial | Created security policy and scanning infrastructure |

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

---

## Web Scraping Dependencies - Security Audit

### New Dependencies Added

The following web scraping dependencies have been added to `requirements.in`:

- **beautifulsoup4>=4.12.0** - HTML/XML parser for web scraping
- **requests>=2.31.0** - HTTP library for making requests
- **schedule>=1.2.0** - Job scheduling for automated scraping
- **lxml>=4.9.0** - Fast XML/HTML parser (used by BeautifulSoup)

### Required Actions

**IMPORTANT:** After adding these dependencies, you MUST:

1. **Regenerate requirements.txt with hashes:**
   ```bash
   # Ensure pip-tools is installed
   pip install pip-tools
   
   # Generate requirements.txt with security hashes
   pip-compile --generate-hashes requirements.in
   
   # This will:
   # - Resolve all dependencies and sub-dependencies
   # - Add SHA256 hashes for verification
   # - Ensure reproducible builds
   ```

2. **Install the updated dependencies:**
   ```bash
   # Sync your environment with the new requirements
   pip-sync requirements.txt
   ```

3. **Run security audit on new dependencies:**
   ```bash
   # Scan for known vulnerabilities
   pip-audit -r requirements.txt
   
   # Expected output: Vulnerabilities found report (if any)
   # Document any findings in the "Dependency Vulnerability Findings" section above
   ```

4. **Review and document findings:**
   - If vulnerabilities are found, assess their impact on this project
   - For critical/high severity: upgrade to patched versions
   - For medium/low severity: evaluate risk and document acceptance if needed
   - Update the "Active Vulnerabilities" section above with findings

5. **Commit the changes:**
   ```bash
   git add requirements.in requirements.txt SECURITY.md
   git commit -m "feat: Add web scraping dependencies with security audit"
   ```

### Security Considerations for Web Scraping

When using these dependencies for web scraping:

**Network Security:**
- Always validate SSL certificates (avoid `verify=False`)
- Use HTTPS when available
- Implement proper timeout values
- Rotate user agents responsibly
- Respect robots.txt and rate limits

**Data Security:**
- Sanitize scraped data before storage
- Never commit scraped data containing PII
- Be cautious with eval() or exec() on scraped content
- Validate and escape HTML before rendering

**Dependency-Specific Risks:**
- **requests**: Known for occasional CVEs; keep updated
- **lxml**: C-based parser; vulnerabilities can have memory implications
- **beautifulsoup4**: Generally safe but depends on underlying parser (lxml)
- **schedule**: Minimal attack surface; low risk

**Monitoring:**
- Regularly update dependencies (monthly minimum)
- Subscribe to security advisories for these packages
- Re-run pip-audit after any dependency updates

### Compliance Notes

- Ensure scraped data complies with website Terms of Service
- Respect GDPR/CCPA if scraping personal data
- Implement proper rate limiting to avoid DoS concerns
- Document legal review of target sites if applicable

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
