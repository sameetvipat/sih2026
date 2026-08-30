#!/usr/bin/env python
"""Fail if the report still contains unfilled placeholders.

The report is drafted with every number as a loud {{MARKER}} rather than a
current value, because the current values were measured with a feature-encoding
bug and will change. A plausible-looking stale number is the easiest way for a
wrong figure to reach a submission, so this makes "forgot to fill one" a build
failure instead of something a reader discovers.
"""
from __future__ import annotations

import re
import sys

PATH = "reports/report.md"


def main() -> int:
    try:
        text = open(PATH).read()
    except FileNotFoundError:
        print(f"{PATH} not found")
        return 1

    holes = sorted(set(re.findall(r"\{\{[A-Z_0-9]+\}\}", text)))
    todos = re.findall(r"\[(?:STATE|TODO|FIXME)[^\]]*\]", text)

    if not holes and not todos:
        print(f"{PATH}: no unfilled placeholders")
        return 0

    print(f"{PATH} is NOT ready: {len(holes)} placeholders, {len(todos)} directives",
          file=sys.stderr)
    for h in holes:
        print(f"  {h}", file=sys.stderr)
    for t in todos:
        print(f"  {t[:70]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
