from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from house_price_predictor.modeling import TARGET_COLUMN


def parse_data_description(path: Path) -> dict[str, dict[str, Any]]:
    # Parse Kaggle data_description.txt: per feature, a short description and optional category code → label list.
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    features: dict[str, dict[str, Any]] = {}
    current: str | None = None
    header_re = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)$")
    category_re = re.compile(r"^\s+(\S+)\s+(.+?)\s*$")

    for line in lines:
        header = header_re.match(line)
        if header:
            name, desc = header.group(1), header.group(2).strip()
            current = name
            features[current] = {"description": desc, "categories": []}
            continue
        if current is None:
            continue
        cat = category_re.match(line)
        if cat and line.strip():
            code, label = cat.group(1), cat.group(2).strip()
            features[current]["categories"].append({"code": code, "label": label})

    for meta in features.values():
        if not meta["categories"]:
            meta.pop("categories", None)
        else:
            meta["kind"] = "categorical"
    return features


def _column_kind(series: pd.Series) -> str:
    # Classify a pandas column as numeric or categorical for documentation and schema output.
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _label_for_training_code(code: str, doc: dict[str, Any]) -> str:
    # Prefer data_description glossary for this code; otherwise explain using the feature description text.
    desc = (doc.get("description") or "").strip()
    cats = doc.get("categories") or []
    for item in cats:
        if str(item.get("code", "")).strip() == str(code).strip():
            return str(item.get("label", "")).strip()
    if desc:
        return f'Training-set code «{code}». {desc}'
    return f'Training-set code «{code}».'


def build_prediction_input_schema(
    *,
    train_path: Path | None,
    test_path: Path | None,
    description_path: Path,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    # Build JSON for POST /predict: per feature, name/kind/description; categoricals get values from train only {code, label}.
    docs = parse_data_description(description_path)
    notes = [
        f"Do not send the target column `{target_column}`.",
        "Each `records` item is one row; keys must match training columns (except the target).",
        "Missing keys or nulls are imputed using training statistics (median / mode).",
        "For categoricals, `values` lists only codes that appear in **train.csv** — send one of those strings (or rely on imputation).",
        "If you upgrade scikit-learn, retrain: `python src/train.py`.",
    ]

    train_exists = train_path is not None and train_path.exists()
    ref_path = train_path if train_exists else (test_path if test_path and test_path.exists() else None)

    if ref_path is None:
        features_out: list[dict[str, Any]] = []
        for name, meta in sorted(docs.items()):
            if name == target_column:
                continue
            entry: dict[str, Any] = {
                "name": name,
                "kind": meta.get("kind", "numeric"),
                "description": meta.get("description", ""),
            }
            if meta.get("kind") == "categorical" or meta.get("categories"):
                entry["values"] = []
            features_out.append(entry)
        extra = [
            "Add `data/raw/train.csv` to populate categorical `values` from your training split only.",
        ]
        if not docs:
            extra.append("Add `data/raw/data_description.txt` for feature descriptions.")
        return {
            "source": "no_csv",
            "table_path": None,
            "train_path": str(train_path) if train_path else None,
            "features": features_out,
            "notes": notes + extra,
            "cheatsheet": "GET /cheatsheet",
        }

    ref = pd.read_csv(ref_path, low_memory=False)
    cols = [c for c in ref.columns if c != target_column]

    train_df: pd.DataFrame | None = None
    if train_exists:
        train_df = pd.read_csv(train_path, low_memory=False)

    features_out: list[dict[str, Any]] = []

    for col in cols:
        s_ref = ref[col]
        kind = _column_kind(s_ref)
        doc = docs.get(col, {})
        entry: dict[str, Any] = {
            "name": col,
            "kind": kind,
            "description": doc.get("description", ""),
        }

        if kind == "categorical" and train_df is not None:
            s_train = train_df[col]
            observed = sorted(s_train.dropna().astype(str).unique().tolist())
            values_out: list[dict[str, str]] = []
            max_n = 500
            for code in observed[:max_n]:
                values_out.append(
                    {
                        "code": code,
                        "label": _label_for_training_code(code, doc),
                    }
                )
            entry["values"] = values_out
            if len(observed) > max_n:
                entry["values_truncated"] = True
        elif kind == "categorical" and train_df is None:
            entry["values"] = []
            entry["values_note"] = "train.csv not available — add it to list allowed training codes."

        features_out.append(entry)

    return {
        "source": "train_values_for_categoricals" if train_df is not None else "columns_from_fallback_csv",
        "table_path": str(ref_path),
        "train_path": str(train_path) if train_path else None,
        "features": features_out,
        "notes": notes,
        "cheatsheet": "GET /cheatsheet",
    }


def build_input_cheatsheet_markdown(
    *,
    train_path: Path | None,
    test_path: Path | None,
    description_path: Path,
    target_column: str = TARGET_COLUMN,
) -> str:
    # Markdown memo aligned with GET /schema: training-only categorical codes with code + label.
    schema = build_prediction_input_schema(
        train_path=train_path,
        test_path=test_path,
        description_path=description_path,
        target_column=target_column,
    )
    lines = [
        "# House Price Predictor — input cheat sheet",
        "",
        "## How to call the API",
        "",
        "Send **POST** `/predict` with JSON:",
        "",
        "```json",
        '{ "records": [ { "Id": 1461, "MSSubClass": 60, "...": "..." } ] }',
        "```",
        "",
        f"- Do **not** include `{target_column}`.",
        "- Property names must match `train.csv` headers.",
        "- **Predictions** are rounded to whole US dollars.",
        "",
        "## General notes",
        "",
    ]
    for note in schema.get("notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "---", "", "## Features (column order from reference CSV)", ""])

    for feat in schema["features"]:
        name = feat["name"]
        kind = feat["kind"]
        lines.append(f"## `{name}` — {kind}")
        lines.append("")
        desc = feat.get("description") or "*No description in data_description.txt.*"
        lines.append(desc)
        lines.append("")

        if kind == "categorical":
            vals = feat.get("values") or []
            if not vals and feat.get("values_note"):
                lines.append(f"*{feat['values_note']}*")
                lines.append("")
                continue
            lines.append("**Allowed values (from training data only)** — send the `code` string:")
            lines.append("")
            for v in vals:
                lines.append(f"- `{v['code']}` — {v['label']}")
            if feat.get("values_truncated"):
                lines.append("")
                lines.append("*List truncated; see GET /schema for full JSON.*")
            lines.append("")
        else:
            lines.append("*Numeric: use a number consistent with training data (see description for units).*")
            lines.append("")

    return "\n".join(lines)
