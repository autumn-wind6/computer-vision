from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_csv(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with open(source, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"path": str(source), "rows": 0, "columns": [], "metrics": {}}

    metrics: dict[str, dict[str, float]] = {}
    for column in rows[0].keys():
        values = [_to_float(row.get(column, "")) for row in rows]
        numeric = [v for v in values if v is not None]
        if not numeric:
            continue
        metrics[column] = {
            "first": numeric[0],
            "last": numeric[-1],
            "min": min(numeric),
            "max": max(numeric),
        }
    return {"path": str(source), "rows": len(rows), "columns": list(rows[0].keys()), "metrics": metrics}


def summaries_to_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| file | rows | metric | first | last | min | max |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        name = Path(summary["path"]).name
        rows = summary["rows"]
        for metric, stats in summary["metrics"].items():
            lines.append(
                f"| {name} | {rows} | {metric} | "
                f"{stats['first']:.6g} | {stats['last']:.6g} | {stats['min']:.6g} | {stats['max']:.6g} |"
            )
    return "\n".join(lines) + "\n"
