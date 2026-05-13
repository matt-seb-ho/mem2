from __future__ import annotations

from html import escape
from typing import Any

ARC_COLORS: dict[int, str] = {
    0: "#111111",
    1: "#1E93FF",
    2: "#F94B3B",
    3: "#4CDC4C",
    4: "#FFDC00",
    5: "#888888",
    6: "#E53AAB",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}

LIGHT_TEXT = {4, 8}


def validate_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise ValueError("Grid must be a non-empty list of rows")
    width = None
    normalized: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list) or not row:
            raise ValueError("Grid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("Grid must be rectangular")
        normalized_row: list[int] = []
        for value in row:
            if not isinstance(value, int) or value not in ARC_COLORS:
                raise ValueError(f"Grid cell must be an integer 0..9, got {value!r}")
            normalized_row.append(value)
        normalized.append(normalized_row)
    return normalized


def grid_shape(grid: Any) -> str:
    normalized = validate_grid(grid)
    return f"{len(normalized)}x{len(normalized[0])}"


def render_grid_html(grid: Any, *, cell_px: int = 18, show_values: bool = False, label: str | None = None) -> str:
    normalized = validate_grid(grid)
    cell_style = (
        f"width:{cell_px}px;height:{cell_px}px;"
        "border:1px solid #666;padding:0;text-align:center;"
        f"line-height:{cell_px}px;font:bold 10px monospace;"
    )
    rows = []
    for row in normalized:
        cells = []
        for value in row:
            color = ARC_COLORS[value]
            text_color = "#222" if value in LIGHT_TEXT else "white"
            content = str(value) if show_values else ""
            cells.append(f"<td style='{cell_style}background:{color};color:{text_color};'>{content}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = "<table style='border-collapse:collapse;display:inline-block;margin:4px;vertical-align:middle;'>" + "".join(rows) + "</table>"
    if not label:
        return table
    safe_label = escape(label)
    return (
        "<div style='display:inline-flex;flex-direction:column;align-items:center;gap:4px;margin:4px;'>"
        f"<span style='font:11px monospace;opacity:0.75;'>{safe_label} ({grid_shape(normalized)})</span>"
        f"{table}</div>"
    )


def render_palette_legend(*, cell_px: int = 22) -> str:
    row = []
    for value in range(10):
        color = ARC_COLORS[value]
        text_color = "#222" if value in LIGHT_TEXT else "white"
        row.append(
            "<td style='"
            f"width:{cell_px}px;height:{cell_px}px;background:{color};"
            "text-align:center;border:1px solid #666;padding:0;"
            f"line-height:{cell_px}px;color:{text_color};font:bold 11px monospace;"
            f"'>{value}</td>"
        )
    return "<table style='border-collapse:collapse;display:inline-block;margin:4px;'><tr>" + "".join(row) + "</tr></table>"


def render_grid_ascii(grid: Any) -> str:
    normalized = validate_grid(grid)
    return "\n".join(" ".join(str(value) for value in row) for row in normalized)


def render_grid_markdown(grid: Any, *, label: str | None = None, cell_px: int = 18) -> str:
    title = f"**{label}**\n\n" if label else ""
    return (
        f"{title}{render_grid_html(grid, cell_px=cell_px, label=label)}\n\n"
        "<details><summary>ASCII fallback</summary>\n\n"
        "```text\n"
        f"{render_grid_ascii(grid)}\n"
        "```\n\n"
        "</details>"
    )


def render_pair_markdown(pair: dict[str, Any], *, label: str, cell_px: int = 18) -> str:
    blocks = []
    if "input" in pair:
        blocks.append(render_grid_html(pair["input"], cell_px=cell_px, label=f"{label} input"))
    if "output" in pair:
        blocks.append(render_grid_html(pair["output"], cell_px=cell_px, label=f"{label} output"))
    if not blocks:
        return ""
    html = "<div style='display:flex;align-items:flex-start;flex-wrap:wrap;gap:10px;margin:8px 0;'>" + "".join(blocks) + "</div>"
    ascii_parts = []
    if "input" in pair:
        ascii_parts.extend([f"{label} input:", render_grid_ascii(pair["input"])])
    if "output" in pair:
        ascii_parts.extend([f"{label} output:", render_grid_ascii(pair["output"])])
    return (
        f"{html}\n\n"
        "<details><summary>ASCII fallback</summary>\n\n"
        "```text\n"
        + "\n\n".join(ascii_parts)
        + "\n```\n\n"
        "</details>"
    )
