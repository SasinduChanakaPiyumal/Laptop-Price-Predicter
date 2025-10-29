"""
Vectorized storage feature extraction utilities.

This module provides fast, vectorized functions to parse the 'Memory'
column from the laptop price dataset and derive useful features like:
- Has_SSD, Has_HDD, Has_Flash, Has_Hybrid (0/1 flags)
- Storage_Capacity_GB (total capacity in GB, summing multiple drives)
- Storage_Type_Score (weighted score by storage type)

It replaces slow per-row apply/regex logic with pandas vectorized
operations that scale better and use less Python overhead.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict

__all__ = [
    "extract_storage_features_vectorized",
    "extract_storage_features_apply",
]


def _sum_capacity_gb(memory: pd.Series) -> pd.Series:
    """
    Compute total storage capacity in GB per row by extracting all TB and GB
    values from the memory string and summing them.
    """
    # Ensure string dtype and handle NaN
    s = memory.fillna("").astype(str)

    # Extract all TB values (as strings), convert to float, sum per row
    tb = (
        s.str.extractall(r"(\d+(?:\.\d+)?)\s*TB", flags=re.IGNORECASE)
        .astype(float)
        .groupby(level=0)
        .sum()
        .squeeze()
    )

    # Extract all GB values (as strings), convert to float, sum per row
    gb = (
        s.str.extractall(r"(\d+(?:\.\d+)?)\s*GB", flags=re.IGNORECASE)
        .astype(float)
        .groupby(level=0)
        .sum()
        .squeeze()
    )

    # Align indices with original series and fill missing with 0
    tb = tb.reindex(s.index, fill_value=0.0)
    gb = gb.reindex(s.index, fill_value=0.0)

    # Convert TB to GB and sum
    total_gb = tb * 1024.0 + gb
    return total_gb


def extract_storage_features_vectorized(memory: pd.Series) -> pd.DataFrame:
    """
    Vectorized extraction of storage features from a pandas Series of strings.

    Returns a DataFrame with columns:
      - Has_SSD, Has_HDD, Has_Flash, Has_Hybrid (int8 0/1)
      - Storage_Capacity_GB (float32)
      - Storage_Type_Score (float32)
    """
    s = memory.fillna("").astype(str)

    # Binary flags using fast vectorized contains
    has_ssd = s.str.contains("SSD", case=False, regex=False).astype("int8")
    has_hdd = s.str.contains("HDD", case=False, regex=False).astype("int8")
    has_flash = s.str.contains("Flash", case=False, regex=False).astype("int8")
    has_hybrid = s.str.contains("Hybrid", case=False, regex=False).astype("int8")

    # Capacity in GB
    total_capacity_gb = _sum_capacity_gb(s).astype("float32")

    # Weighted score
    storage_type_score = (
        has_ssd.astype("int16") * 3
        + has_flash.astype("int16") * 25 // 10  # 2.5 approximated as 2.5 -> 2.5 float below
        + has_hybrid.astype("int16") * 2
        + has_hdd.astype("int16") * 1
    ).astype("float32")

    # Use precise float weights (avoiding int truncation above)
    storage_type_score = (
        has_ssd.astype("float32") * 3.0
        + has_flash.astype("float32") * 2.5
        + has_hybrid.astype("float32") * 2.0
        + has_hdd.astype("float32") * 1.0
    )

    df = pd.DataFrame(
        {
            "Has_SSD": has_ssd,
            "Has_HDD": has_hdd,
            "Has_Flash": has_flash,
            "Has_Hybrid": has_hybrid,
            "Storage_Capacity_GB": total_capacity_gb,
            "Storage_Type_Score": storage_type_score.astype("float32"),
        },
        index=s.index,
    )
    return df


# Reference implementation (slow) kept for benchmarking correctness

def extract_storage_features_apply(memory: pd.Series) -> pd.DataFrame:
    """
    Original apply-based reference implementation for correctness and benchmarking.
    """
    def _extract_one(memory_string: str):
        memory_string = str(memory_string)
        has_ssd = 0
        has_hdd = 0
        has_flash = 0
        has_hybrid = 0
        total_capacity_gb = 0.0

        if "SSD" in memory_string:
            has_ssd = 1
        if "HDD" in memory_string:
            has_hdd = 1
        if "Flash" in memory_string:
            has_flash = 1
        if "Hybrid" in memory_string:
            has_hybrid = 1

        tb_matches = re.findall(r"(\d+(?:\.\d+)?)\s*TB", memory_string)
        gb_matches = re.findall(r"(\d+(?:\.\d+)?)\s*GB", memory_string)
        for tb in tb_matches:
            total_capacity_gb += float(tb) * 1024
        for gb in gb_matches:
            total_capacity_gb += float(gb)

        return has_ssd, has_hdd, has_flash, has_hybrid, total_capacity_gb

    tuples = memory.apply(_extract_one)
    df = pd.DataFrame(
        {
            "Has_SSD": tuples.apply(lambda x: x[0]).astype("int8"),
            "Has_HDD": tuples.apply(lambda x: x[1]).astype("int8"),
            "Has_Flash": tuples.apply(lambda x: x[2]).astype("int8"),
            "Has_Hybrid": tuples.apply(lambda x: x[3]).astype("int8"),
            "Storage_Capacity_GB": tuples.apply(lambda x: x[4]).astype("float32"),
        }
    )
    df["Storage_Type_Score"] = (
        df["Has_SSD"].astype("float32") * 3.0
        + df["Has_Flash"].astype("float32") * 2.5
        + df["Has_Hybrid"].astype("float32") * 2.0
        + df["Has_HDD"].astype("float32") * 1.0
    ).astype("float32")
    return df
