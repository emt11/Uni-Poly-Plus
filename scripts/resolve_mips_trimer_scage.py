#!/usr/bin/env python3
"""Fail-closed placeholder until the next MTS configuration is defined.

The retired T/G/R/A configuration schema is intentionally not parsed or
translated. This command is side-effect free: it does not inspect the cache,
create output paths, import training code, or emit shell exports.
"""

from __future__ import annotations

import sys


MESSAGE = (
    "No active MTS configuration schema; define the next configuration "
    "before launching production MTS."
)


def main(argv: list[str] | None = None) -> int:
    # Any positional path or legacy option is rejected identically. Keeping
    # argv opaque prevents an old field name from becoming a compatibility
    # branch by accident.
    _ = argv if argv is not None else sys.argv[1:]
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
