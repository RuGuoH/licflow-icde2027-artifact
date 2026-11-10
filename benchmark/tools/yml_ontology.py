"""从 licenses_description.yml 读取人工 ontology（S7-M replay gold）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from common import REPO_ROOT


def load_yml_ontology(short_id: str) -> dict[str, Any] | None:
    """按 short_id 返回 licenses_description.yml 中单条许可 dict。"""
    import yaml

    yml_path = REPO_ROOT / "scripts" / "licenses_description.yml"
    if not yml_path.is_file():
        return None
    data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        return None
    target = (short_id or "").strip()
    if not target:
        return None
    low = target.casefold()
    for entry in data[1:]:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("short_id", ""))
        if sid == target or sid.casefold() == low:
            return entry
    return None


def resolve_ontology_gold_ref(ref: str | None) -> dict[str, Any] | None:
    """解析 experiment.ontology_gold_ref，如 ``yml:MIT``。"""
    if not ref or not str(ref).startswith("yml:"):
        return None
    short_id = str(ref).split(":", 1)[1].strip()
    return load_yml_ontology(short_id)
