from __future__ import annotations

import difflib
from typing import List


def diff_markdown(a: str, b: str) -> str:
    """Return a markdown-friendly diff between two strings."""

    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    if not diff:
        return "No differences detected."

    # Wrap in a fenced code block with diff syntax highlighting.
    return "\n".join(["```diff"] + diff + ["```"])


def side_by_side_diff(a: str, b: str, width: int = 80) -> str:
    """Generate a simple side-by-side diff in plain text."""

    a_lines = a.splitlines()
    b_lines = b.splitlines()
    differ = difflib.Differ()
    diff_lines = list(differ.compare(a_lines, b_lines))
    return "\n".join(diff_lines)
