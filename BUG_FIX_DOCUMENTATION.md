# Bug Fix Documentation

## Bug Description

**Location:** `Laptop Price model(1).py`, lines 160-168 (function `set_processor`)

**Type:** IndexError

**Severity:** High - causes runtime crash

### The Problem

The original `set_processor` function had a bug that would raise an `IndexError` when processing an empty string:

```python
def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':  # BUG HERE
            return 'AMD'
        else:
            return 'Other'
```

When `name` is an empty string `""`, the expression `name.split()` returns an empty list `[]`. Attempting to access `name.split()[0]` on an empty list raises an `IndexError: list index out of range`.

### When This Bug Could Occur

This bug could manifest in several scenarios:
1. If the dataset contains empty strings in the CPU name column
2. If data preprocessing results in empty strings
3. If the function is called with invalid/missing data

### The Fix

The fixed version checks if the split result is non-empty before accessing the first element:

```python
def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        split_name = name.split()
        if split_name and split_name[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
```

**Changes made:**
1. Store the result of `name.split()` in a variable `split_name`
2. Check if `split_name` is non-empty using `if split_name and ...`
3. Only access `split_name[0]` if the list is not empty

This ensures that empty strings (or strings containing only whitespace) are properly handled and categorized as 'Other' instead of causing a crash.

## Unit Tests

A comprehensive unit test suite has been created in `test_laptop_price_model.py` that:

1. **Tests the fixed function works correctly** for:
   - Intel Core i7, i5, i3 processors
   - AMD processors
   - Other processor types
   - Empty strings (the bug case)
   - Whitespace-only strings

2. **Demonstrates the original bug** by showing that the original version raises an `IndexError` on empty strings

### Running the Tests

```bash
python test_laptop_price_model.py
```

Expected output:
```
........
----------------------------------------------------------------------
Ran 8 tests in 0.XXXs

OK
```

All tests should pass with the fixed version. The test `test_original_fails_on_empty_string` specifically demonstrates that the original buggy version would fail with an `IndexError`.

## Impact

- **Before the fix:** The script would crash with an `IndexError` if any empty CPU names were encountered
- **After the fix:** Empty CPU names are gracefully handled and categorized as 'Other'
- **No behavioral change** for valid, non-empty CPU names - all existing functionality is preserved
