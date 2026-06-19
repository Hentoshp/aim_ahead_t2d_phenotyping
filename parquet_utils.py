from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    import pyarrow
except ImportError:  # pragma: no cover - pyarrow is part of the project env
    pyarrow = None


PARQUET_COMPAT_ERROR = "Repetition level histogram size mismatch"


def read_parquet_with_compat_hint(path: str | Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, **kwargs)
    except OSError as exc:
        if PARQUET_COMPAT_ERROR not in str(exc):
            raise

        pyarrow_version = pyarrow.__version__ if pyarrow is not None else "not-installed"
        raise RuntimeError(
            "Failed to read parquet data in the current Python environment. "
            f"File: {Path(path)}. This usually means the runtime is using an incompatible "
            "pandas/pyarrow stack for parquet files produced by this project. "
            "Use the project's conda environment, for example `conda activate aim_ahead`, "
            "then rerun the pipeline step.\n"
            f"Current environment: Python {sys.version.split()[0]}, "
            f"pandas {pd.__version__}, pyarrow {pyarrow_version}."
        ) from exc
