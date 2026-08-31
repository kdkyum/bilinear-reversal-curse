# Load and merge ../results/lre/*.json into a single pandas DataFrame
# - Supports JSON object, JSON array of objects, and JSONL (one JSON per line)
# - Adds a helper column _source_file to identify the origin of each record (we'll drop it below as requested)
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any


def _coerce_record(x: Any) -> Dict[str, Any]:
    """Coerce any JSON-parsed value into a dict row for DataFrame consumption."""
    if isinstance(x, dict):
        return x
    # Wrap non-dict values in a consistent structure
    return {"_value": x}


def load_results(base_dir: str = "../results/lre", pattern: str = "*.json") -> pd.DataFrame:
    base = Path(base_dir)
    files = sorted(base.glob(pattern))
    rows: List[Dict[str, Any]] = []

    if not files:
        print(f"No files matched: {base_dir}/{pattern}")
        return pd.DataFrame()

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: read error: {e}")
            continue

        if not text:
            print(f"[WARN] Skipping {fp.name}: file is empty")
            continue

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try JSON Lines (each non-empty line is a JSON object/value)
            recs = []
            ok = False
            for i, line in enumerate(text.splitlines(), start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    recs.append(json.loads(s))
                    ok = True
                except json.JSONDecodeError:
                    print(f"[WARN] {fp.name}: line {i} is not valid JSON; skipping this line")
            if ok:
                for r in recs:
                    row = _coerce_record(r)
                    row.setdefault("_source_file", fp.name)
                    rows.append(row)
            else:
                print(f"[WARN] Skipping {fp.name}: not valid JSON/JSONL")
            continue

        # Handle normal JSON
        if isinstance(parsed, list):
            for r in parsed:
                row = _coerce_record(r)
                row.setdefault("_source_file", fp.name)
                rows.append(row)
        elif isinstance(parsed, dict):
            row = _coerce_record(parsed)
            row.setdefault("_source_file", fp.name)
            rows.append(row)
        else:
            # Primitive at top-level
            row = {"_value": parsed, "_source_file": fp.name}
            rows.append(row)

    if not rows:
        print("No records parsed from matched files.")
        return pd.DataFrame()

    # json_normalize helps if some records contain nested objects
    df = pd.json_normalize(rows, sep=".")
    if not df.empty and "_source_file" in df.columns:
        df = df.drop(columns=["_source_file"], errors="ignore")
    return df