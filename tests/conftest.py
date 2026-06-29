"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.generate_synthetic_data import generate
from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """A small, schema-correct synthetic frame for fast unit tests."""
    return generate(n_rows=300, seed=7)
