"""Backwards-compatible shim for notebooks/.

Real implementation lives at ``src/ingest/load_fda_shortages.py``.
Run instead via:  python -m ingest.load_fda_shortages
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ingest.load_fda_shortages import main  # noqa: E402, F401


if __name__ == "__main__":
    main()
