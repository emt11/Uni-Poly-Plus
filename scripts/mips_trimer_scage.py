#!/usr/bin/env python3
"""Fail-closed placeholder until the next MTS configuration is defined."""

import sys


def main(argv=None):
    _ = argv
    print(
        "No active MTS configuration schema; define the next configuration "
        "before launching production MTS.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
