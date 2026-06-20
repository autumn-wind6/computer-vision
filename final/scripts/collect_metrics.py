#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvfinal.metrics import summaries_to_markdown, summarize_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect training/evaluation CSV metrics for the report.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summaries = [summarize_csv(path) for path in args.inputs]
    json_target = Path(args.output_json)
    md_target = Path(args.output_md)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_target.write_text(summaries_to_markdown(summaries), encoding="utf-8")
    print(json_target)
    print(md_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
