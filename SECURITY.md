# Security Policy

## Backlog Items

### Artifact History Management

**Status**: Open
**Priority**: Medium
**Created**: Current date

#### Issue
The repository's `predictor.pickle` file (machine learning model artifact) is referenced in the codebase and may have been committed to git history in the past. While this file is now excluded via `.gitignore` at HEAD, historical commits may still contain it.

#### Decision
**History rewrite deferred** for the following reasons:
1. The pickle file contains a trained ML model, not sensitive credentials or PII
2. History rewriting requires force-push which may disrupt collaborators
3. The file is now properly ignored going forward
4. Risk assessment: LOW - model artifacts are typically not security-sensitive unless they contain embedded training data with PII

#### Compensating Controls
- `.gitignore` now excludes all `.pkl`, `.pickle`, and `predictor.pickle` files
- Git commands have been documented to remove these files from current HEAD:
  ```bash
  git rm -r --cached --ignore-unmatch predictor.pickle
  git rm -r --cached --ignore-unmatch '*.pkl' '*.pickle'
  ```
- Future model artifacts will not be tracked

#### Follow-up Actions
- [ ] If this repository will be made public, reassess the need for history rewrite
- [ ] Consider using Git LFS or artifact storage for large model files in the future
- [ ] If predictor.pickle is found to contain sensitive information, escalate to immediate history rewrite using `git filter-repo` or BFG Repo-Cleaner

#### References
- ML_IMPROVEMENTS_SUMMARY.md documents the purpose of predictor.pickle
- Codebase references: `Laptop Price model(1).py:519`, `Laptop Price model.ipynb`
