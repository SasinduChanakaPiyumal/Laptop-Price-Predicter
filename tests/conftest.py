import os
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def test_data():
    """Load the small representative test dataset as a fresh copy per test.

    Tries UTF-8 first, then falls back to latin-1 to mirror production behavior
    for the original laptop_price.csv which may contain extended characters.
    """
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "test_data.csv")

    encodings_to_try = ["utf-8", "latin-1", "iso-8859-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            # Return a defensive copy to avoid cross-test contamination
            return df.copy()
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # If we get here, all attempted encodings failed
    raise UnicodeDecodeError(
        last_err.encoding if hasattr(last_err, "encoding") else "utf-8",
        last_err.object if hasattr(last_err, "object") else b"",
        getattr(last_err, "start", 0),
        getattr(last_err, "end", 0),
        getattr(last_err, "reason", str(last_err)),
    )
