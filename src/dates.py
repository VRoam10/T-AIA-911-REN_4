# dates.py
from __future__ import annotations

from typing import List

import dateparser


def normalize_dates_fr(date_strings: List[str]) -> List[str]:
    out = []
    for s in date_strings:
        dt = dateparser.parse(
            s,
            languages=["fr"],
            settings={"PREFER_DATES_FROM": "future", "DATE_ORDER": "DMY"},
        )
        if dt:
            out.append(dt.date().isoformat())

    # dedupe keep order
    seen = set()
    uniq = []
    for d in out:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq
