# Contributing to Laptop Price Prediction Model

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Security](#security)

## 🤝 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat everyone with respect and consideration
- **Be collaborative**: Work together constructively
- **Be inclusive**: Welcome diverse perspectives and experiences
- **Be professional**: Focus on what is best for the project
- **Be patient**: Remember that everyone has different experience levels

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/laptop-price-prediction.git
   cd laptop-price-prediction
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/laptop-price-prediction.git
   ```
4. **Set up your development environment** (see below)

## 💻 Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip and pip-tools
- Git
- (Optional) virtualenv or conda

### Step-by-Step Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Activate on Linux/macOS
   source .venv/bin/activate
   
   # Activate on Windows
   .venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   # Install pip-tools for dependency management
   pip install pip-tools
   
   # Generate requirements.txt with hashes
   pip-compile --generate-hashes -o requirements.txt requirements.in
   
   # Install dependencies
   pip-sync requirements.txt
   
   # Install development tools
   pip install pip-audit pytest black flake8 mypy
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, xgboost; print('Setup successful!')"
   ```

4. **Run security scans** (recommended):
   ```bash
   # Scan dependencies for vulnerabilities
   pip-audit -r requirements.txt
   ```

### Dataset Setup

Ensure you have `laptop_price.csv` in the project root. If you're working with a different dataset:
- Document the source and license
- Update the README with new dataset information
- Ensure Latin-1 or UTF-8 encoding

## 🎯 How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **🐛 Bug Fixes**: Fix issues with existing code
2. **✨ New Features**: Add new ML models, feature engineering techniques, or functionality
3. **📚 Documentation**: Improve README, docstrings, or create tutorials
4. **🔬 Performance Improvements**: Optimize algorithms or reduce training time
5. **🧪 Testing**: Add or improve test coverage
6. **🎨 Code Quality**: Refactoring, type hints, or code style improvements

### Contribution Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following the coding standards (see below)

3. **Test your changes**:
   ```bash
   # Run the main script
   python "Laptop Price model(1).py"
   
   # Check for syntax errors
   python -m py_compile "Laptop Price model(1).py"
   ```

4. **Commit your changes** with a descriptive message (see commit guidelines)

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guide
- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **88 characters** (Black default) or 79 (PEP 8)
- Use **descriptive variable names** (e.g., `storage_capacity_gb` not `sc`)

### Code Formatting

We recommend using **Black** for automatic code formatting:

```bash
# Install Black
pip install black

# Format your code
black "Laptop Price model(1).py"

# Check formatting without applying
black --check "Laptop Price model(1).py"
```

### Type Hints

Add type hints where possible to improve code clarity:

```python
def extract_storage_features(memory_string: str) -> tuple[int, int, int, int, float]:
    """
    Extract storage features from memory string.
    
    Args:
        memory_string: Memory specification string
        
    Returns:
        Tuple of (has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb)
    """
    # Implementation
    pass
```

### Docstrings

Use **Google-style docstrings** for all functions and classes:

```python
def model_acc(model, model_name: str = "Model", use_scaled: bool = False):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained sklearn model to evaluate
        model_name: Display name for the model
        use_scaled: If True, use scaled data (for linear models)
        
    Returns:
        Tuple of (r2_score, mae, rmse, cv_mean)
        
    Example:
        >>> r2, mae, rmse, cv = model_acc(rf_model, "Random Forest")
        >>> print(f"R² Score: {r2:.4f}")
    """
    pass
```

### Code Organization

- **Group imports**: Standard library → Third-party → Local
- **Separate sections**: Use comments and blank lines to separate logical sections
- **Extract functions**: Break down long code blocks into reusable functions
- **Avoid global state**: Use function parameters and return values

Example:
```python
#!/usr/bin/env python
# coding: utf-8

# Standard library imports
import os
import pickle
import tempfile

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports (if any)
# from utils import custom_function

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.25
```

## 🧪 Testing Guidelines

### Manual Testing

Before submitting a PR, ensure:

1. **Script runs without errors**:
   ```bash
   python "Laptop Price model(1).py"
   ```

2. **Output is reasonable**:
   - Models train successfully
   - R² scores are between 0 and 1
   - MAE and RMSE are positive
   - Best model is selected and saved

3. **Edge cases work**:
   - Dataset with missing values
   - Different encoding types
   - Various storage configurations

### Adding Tests (Future Enhancement)

If you're adding pytest tests, place them in a `tests/` directory:

```
tests/
├── __init__.py
├── test_preprocessing.py
├── test_feature_engineering.py
└── test_models.py
```

Example test:
```python
import pytest
import pandas as pd

def test_extract_storage_features():
    """Test storage feature extraction."""
    from main import extract_storage_features
    
    result = extract_storage_features("256GB SSD + 1TB HDD")
    has_ssd, has_hdd, has_flash, has_hybrid, capacity = result
    
    assert has_ssd == 1
    assert has_hdd == 1
    assert capacity == 1280.0  # 256 + 1024
```

## 💬 Commit Message Guidelines

### Format

Follow the **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semi-colons, etc.)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **chore**: Changes to build process or auxiliary tools

### Examples

```bash
# Feature addition
git commit -m "feat(storage): add Flash and Hybrid storage detection"

# Bug fix
git commit -m "fix(preprocessing): handle null values in Weight column"

# Documentation
git commit -m "docs(README): update installation instructions"

# Performance improvement
git commit -m "perf(models): reduce RandomizedSearchCV iterations to 40"
```

### Best Practices

- **Use imperative mood**: "add feature" not "added feature"
- **Keep subject line under 50 characters**
- **Add detailed body for complex changes**
- **Reference issues**: Include `Fixes #123` or `Closes #456`
- **Sign commits** (optional but recommended):
  ```bash
  git commit -S -m "feat: add new model"
  ```

## 🔄 Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed functionality
2. **Add yourself to contributors** (if applicable)
3. **Ensure code follows style guidelines**
4. **Run security scans**:
   ```bash
   pip-audit -r requirements.txt
   ```
5. **Rebase on latest main** (if needed):
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
Describe how you tested your changes

## Performance Impact
- Training time: [faster/slower/same]
- Model accuracy: [better/worse/same]
- Memory usage: [more/less/same]

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Security scan passed
- [ ] Commit messages follow guidelines

## Related Issues
Fixes #123
Closes #456
```

### Review Process

1. **Automated checks** will run (if configured)
2. **Maintainer review**: Expect feedback within 2-7 days
3. **Address feedback**: Make requested changes
4. **Approval**: Once approved, a maintainer will merge

### After Merge

1. **Delete your branch** (optional):
   ```bash
   git branch -d feature/your-feature-name
   ```
2. **Update your fork**:
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Try latest version** to see if it's already fixed
3. **Gather information**:
   - Python version: `python --version`
   - Package versions: `pip list`
   - Operating system
   - Error messages and stack traces

### Issue Template

```markdown
## Bug Report / Feature Request

**Type**: [Bug / Feature / Enhancement / Question]

### Description
Clear and concise description

### Steps to Reproduce (for bugs)
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
What you expected to happen

### Actual Behavior
What actually happened

### Environment
- Python version: 3.x.x
- OS: [Windows/Linux/macOS]
- Package versions: (paste output of `pip list`)

### Error Messages
```
Paste error messages and stack traces here
```

### Additional Context
Screenshots, logs, or other relevant information
```

## 📚 Documentation

### What to Document

- **New features**: Add usage examples to README
- **API changes**: Update docstrings and documentation
- **Performance changes**: Document impact on speed/accuracy
- **Breaking changes**: Clearly mark and explain migration path

### Documentation Style

- Use **Markdown** for all documentation
- Include **code examples** where applicable
- Add **diagrams** for complex concepts (Mermaid preferred)
- Keep **line length under 80-100 characters** for readability
- Use **clear headings** and table of contents

### Where to Add Documentation

| Type | Location |
|------|----------|
| Usage instructions | README.md |
| Architecture details | ARCHITECTURE.md |
| ML improvements | IMPROVEMENTS_IMPLEMENTED.md |
| API documentation | Docstrings in code |
| Security info | SECURITY.md |
| Contribution guide | CONTRIBUTING.md (this file) |

## 🔐 Security

### Reporting Security Vulnerabilities

**Do NOT create public issues for security vulnerabilities.**

Instead:
1. Email maintainers directly (see README for contact)
2. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Security Best Practices

When contributing:

1. **Never commit secrets**:
   - No API keys, passwords, or tokens
   - Use environment variables for sensitive data
   - Review changes before committing

2. **Scan dependencies**:
   ```bash
   pip-audit -r requirements.txt
   ```

3. **Keep dependencies updated**:
   - Regularly update `requirements.in`
   - Regenerate `requirements.txt` with hashes
   - Test after updates

4. **Follow secure coding practices**:
   - Validate user input
   - Handle errors gracefully
   - Use safe pickle alternatives for untrusted data

## 🏆 Recognition

Contributors will be recognized in:
- Git commit history
- CONTRIBUTORS.md file (if created)
- Release notes for significant contributions

## 📞 Getting Help

If you need help:

1. **Check documentation**: README, ARCHITECTURE.md, etc.
2. **Search issues**: Someone may have had the same question
3. **Create an issue**: Use the "Question" label
4. **Contact maintainers**: See README for contact information

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to the Laptop Price Prediction Model! 🎉

For questions about this guide, please open an issue with the "documentation" label.
