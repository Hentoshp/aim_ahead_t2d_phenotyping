from pathlib import Path

import pandas as pd
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from parquet_utils import read_parquet_with_compat_hint


def test_read_parquet_with_compat_hint_preserves_non_compat_errors(monkeypatch: pytest.MonkeyPatch):
    def _raise(*args, **kwargs):
        raise OSError("some other parquet problem")

    monkeypatch.setattr(pd, "read_parquet", _raise)
    with pytest.raises(OSError, match="some other parquet problem"):
        read_parquet_with_compat_hint("fake.parquet")


def test_read_parquet_with_compat_hint_raises_actionable_message(monkeypatch: pytest.MonkeyPatch):
    def _raise(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(pd, "read_parquet", _raise)
    with pytest.raises(RuntimeError, match="conda activate aim_ahead"):
        read_parquet_with_compat_hint("fake.parquet")
