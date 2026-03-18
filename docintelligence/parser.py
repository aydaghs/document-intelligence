from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _group_blocks_by_line(blocks: List[Dict[str, Any]], y_tolerance: int = 12) -> List[List[Dict[str, Any]]]:
    """Group OCR result blocks into pseudo-lines based on vertical proximity."""

    # Each block should contain bbox: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    # We'll use the top y coordinate (y0) as the key.
    blocks = [b for b in blocks if b.get("bbox") and b.get("text")]
    # compute median y for each block
    def block_y(block: Dict[str, Any]) -> float:
        return float(block["bbox"][0][1])

    sorted_blocks = sorted(blocks, key=block_y)

    lines: List[List[Dict[str, Any]]] = []
    for block in sorted_blocks:
        y = block_y(block)
        if not lines:
            lines.append([block])
            continue

        # compare to last line's average y
        last_line = lines[-1]
        avg_y = sum(block_y(b) for b in last_line) / len(last_line)
        if abs(y - avg_y) <= y_tolerance:
            last_line.append(block)
        else:
            lines.append([block])

    # sort blocks in each line by x coordinate
    for line in lines:
        line.sort(key=lambda b: float(b["bbox"][0][0]))

    return lines


def _line_text(line: List[Dict[str, Any]]) -> str:
    return " ".join(b.get("text", "").strip() for b in line if b.get("text"))


def _detect_table(lines: List[List[Dict[str, Any]]], column_tolerance: int = 18) -> List[Dict[str, Any]]:
    """Detect simple tables based on column alignment across consecutive lines."""

    tables: List[Dict[str, Any]] = []

    # Build x anchors for each line
    def line_x_anchors(line: List[Dict[str, Any]]) -> List[float]:
        return [float(block["bbox"][0][0]) for block in line]

    # sliding window over lines to find repeated column patterns
    i = 0
    while i < len(lines):
        anchors = line_x_anchors(lines[i])
        if len(anchors) < 2:
            i += 1
            continue

        group = [lines[i]]

        # look ahead for consecutive lines that align on x positions
        for j in range(i + 1, len(lines)):
            next_anchors = line_x_anchors(lines[j])
            if len(next_anchors) != len(anchors):
                break

            aligned = all(abs(a - b) <= column_tolerance for a, b in zip(anchors, next_anchors))
            if not aligned:
                break
            group.append(lines[j])

        # if we found at least 3 rows, consider it a table
        if len(group) >= 3:
            headers = [_line_text(group[0])] if group else []
            rows = [[block.get("text", "").strip() for block in row] for row in group]
            tables.append({"start_line": i, "row_count": len(group), "rows": rows})
            i += len(group)
        else:
            i += 1

    return tables


def _normalize_donut_cell(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, (str, int, float, bool)):
        return str(cell)
    return str(cell)


def _extract_table_from_list(table_obj: List[Any]) -> List[List[str]]:
    """Convert a list of lists/dicts into a normalized table (list of rows)."""

    if not table_obj:
        return []

    if all(isinstance(r, (list, tuple)) for r in table_obj):
        return [[_normalize_donut_cell(c) for c in row] for row in table_obj]

    if all(isinstance(r, dict) for r in table_obj):
        keys = sorted({k for r in table_obj for k in r.keys()})
        rows = [[str(k) for k in keys]]
        for r in table_obj:
            rows.append([_normalize_donut_cell(r.get(k)) for k in keys])
        return rows

    return []


def _collect_tables_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    """Recursively search Donut-parsed output for table-like structures."""

    tables: List[Dict[str, Any]] = []

    if isinstance(parsed, list):
        # list could directly represent a table
        if _extract_table_from_list(parsed):
            rows = _extract_table_from_list(parsed)
            if len(rows) > 1:
                tables.append({"source": "donut", "row_count": len(rows), "rows": rows})

        for item in parsed:
            tables.extend(_collect_tables_from_parsed(item))

    elif isinstance(parsed, dict):
        for value in parsed.values():
            tables.extend(_collect_tables_from_parsed(value))

    return tables


def parse_layout(ocr_data: List[Dict[str, Any]], donut_parsed: Any = None) -> Dict[str, Any]:
    """Parse OCR blocks into structured layout data.

    Returns a dictionary with:
      - text: full extracted text
      - lines: list of lines (strings)
      - tables: list of detected tables (rows)
    """

    blocks = []
    for page in ocr_data:
        if isinstance(page, dict) and "blocks" in page:
            blocks.extend(page["blocks"])
        elif isinstance(page, dict) and "text" in page:
            blocks.append(page)

    lines = _group_blocks_by_line(blocks)
    line_texts = [_line_text(line) for line in lines if _line_text(line).strip()]
    tables = _detect_table(lines)

    # If Donut parsing is available, look for tabular structures in its output.
    if donut_parsed is not None:
        donut_tables = _collect_tables_from_parsed(donut_parsed)
        if donut_tables:
            tables.extend(donut_tables)

    return {
        "text": "\n".join(line_texts),
        "lines": line_texts,
        "tables": tables,
        "raw_blocks": blocks,
    }
