#!/usr/bin/env python3
"""Compatibility shim for the renamed MTS dispatcher.

New automation should call ``scripts/mts.py``.  Keeping this tiny shim avoids
breaking old tmux commands while ensuring there is only one implementation.
"""

from scripts.mts import main


if __name__ == "__main__":
    raise SystemExit(main())
