#!/usr/bin/env python3
"""Extract filenames from autotables in LaTeX files."""

import sys
from typing import Set  # noqa: F401

found = set()  # type: Set[str]

for arg in sys.argv[1:]:
    with open(arg) as infile:
        for line in infile:
            line = line.partition('$ROW')[2]
            if not line:
                continue
            line = line.partition(':')[2]
            for name in line.split():
                if name not in found:
                    found.add(name)
                    print('results/%s.pkl' % name)
