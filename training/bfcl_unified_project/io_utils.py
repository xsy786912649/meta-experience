import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _make_json_serializable(value):
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def write_jsonl(path: Path, entries: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            entry = _make_json_serializable(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
