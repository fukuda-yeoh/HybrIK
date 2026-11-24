#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`hybrik.backend`.

The implementation lives in the package so other projects can import it. This
wrapper keeps ``python3 scripts/detect_torch_backend.py`` working from a fresh
checkout, before HybrIK has been installed into the environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrik.backend import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
