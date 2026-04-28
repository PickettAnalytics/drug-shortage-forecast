"""Backwards-compatible shim for notebooks/.

Real implementation lives at ``src/shortage_forecast/baseline.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shortage_forecast.baseline import *  # noqa: E402, F401, F403
from shortage_forecast.baseline import (  # noqa: E402, F401
    main,
)


if __name__ == "__main__":
    main()
