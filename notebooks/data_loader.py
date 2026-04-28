"""Backwards-compatible shim for notebooks/.

The data loader now lives at ``src/shortage_forecast/data_loader.py``. This
file makes ``from data_loader import ...`` keep working in the existing
.ipynb notebooks by adding ``src/`` to ``sys.path`` and re-exporting the
package's public surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shortage_forecast.data_loader import *  # noqa: E402, F401, F403
from shortage_forecast.data_loader import (  # noqa: E402, F401
    _apply_exclusion,
    _coerce_dtypes,
    _load_raw,
    _print_split_summary,
    _slice_split,
)
