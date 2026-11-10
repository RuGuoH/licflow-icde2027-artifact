"""Benchmark 共享工具：路径、case 加载、EVENT 抽取。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
CASES_DIR = BENCHMARK_ROOT / "cases"
VARIANTS_DIR = BENCHMARK_ROOT / "variants"
GOLD_PROGRAMS_DIR = BENCHMARK_ROOT / "gold" / "programs"
GOLD_EVENTS_DIR = BENCHMARK_ROOT / "gold" / "events"
RESULTS_DIR = BENCHMARK_ROOT / "results"
SCRIPTS_DIR = REPO_ROOT / "scripts"

EVENT_TYPE_BUCKETS = {
    "errors": ("error",),
    "warnings": ("warning",),
    "restrictions": ("restriction",),
    "policy": ("policy",),
}


def ensure_scripts_env() -> None:
    """将 repo root / scripts 加入 sys.path，并 chdir 到 scripts（main_case 兼容）。"""
    import sys

    root = str(REPO_ROOT)
    scripts = str(SCRIPTS_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    os.chdir(SCRIPTS_DIR)


def load_case(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_path(case_id: str) -> Path:
    return CASES_DIR / f"{case_id}.json"


def variant_path(variant_id: str) -> Path:
    return VARIANTS_DIR / f"{variant_id}.json"


def resolve_complete_case_id(case: dict) -> str:
    """返回 gold events / program 对齐用的 complete case ID。"""
    paired = case.get("paired_complete")
    if paired:
        return str(paired)
    vid = str(case.get("id") or "")
    for suffix in ("_partial_ambiguous", "_ambiguous"):
        if vid.endswith(suffix):
            return vid[: -len(suffix)]
    return vid


def load_case_by_id(case_id: str) -> dict:
    """从 cases/ 或 variants/ 加载 case 定义。"""
    p = case_path(case_id)
    if p.is_file():
        return load_case(p)
    vp = variant_path(case_id)
    if vp.is_file():
        return load_case(vp)
    raise FileNotFoundError(f"case/variant 不存在: {case_id}")


def load_complete_case(case: dict) -> dict:
    """加载 variant 对应的 complete case（用于 gold events 查找）。"""
    cid = resolve_complete_case_id(case)
    if cid == case.get("id"):
        return case
    return load_case_by_id(cid)


def gold_program_path(case: dict) -> Path:
    rel = case["gold"]["program_path"]
    return BENCHMARK_ROOT / rel.replace("/", os.sep)


def gold_events_path(case_id: str) -> Path:
    return GOLD_EVENTS_DIR / f"{case_id}.json"


def gold_events_path_for(case: dict, track: str = "default") -> Path:
    """按轨道返回 event gold 路径。track: default | baseline | resolved"""
    gold = case.get("gold") or {}
    if track == "resolved":
        rel = gold.get("engine_events_resolved")
        if rel:
            return BENCHMARK_ROOT / rel.replace("/", os.sep)
    if track in ("baseline", "default"):
        rel = gold.get("engine_events_baseline")
        if rel:
            return BENCHMARK_ROOT / rel.replace("/", os.sep)
    return gold_events_path(case["id"])


def load_gold_events(case: dict, track: str = "default") -> list[dict]:
    lookup = load_complete_case(case)
    path = gold_events_path_for(lookup, track)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("events", [])


def gold_program_path_for(case: dict, *, resolved: bool = False) -> Path:
    gold = case.get("gold") or {}
    if resolved:
        rel = gold.get("program_path_resolved")
        if rel:
            return BENCHMARK_ROOT / rel.replace("/", os.sep)
    return gold_program_path(case)


def normalize_event_info(info: str) -> str:
    """弱化具体 Work 名，便于 Event-F1 匹配。"""
    text = info.strip()
    text = re.sub(r"\$WNAME\$|\$WTYPE\$|\$LNAME\$", "<PH>", text)
    text = re.sub(r"\bWork\s+[A-Za-z0-9_./\-]+", "Work <NAME>", text)
    text = re.sub(r"\bE_[A-Za-z0-9_]+", "E_<NAME>", text)
    text = re.sub(r"\b[ACG]_[A-Za-z0-9_]+", "<COMPOSITE>", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def event_key(event: dict) -> tuple[str, str]:
    return (event.get("type", "").strip().lower(), normalize_event_info(event.get("info", "")))


def events_from_work(work) -> list[dict]:
    out: list[dict] = []
    for ev in getattr(work, "caution", []) or []:
        out.append({"type": ev.type, "info": ev.info})
    return out


def bucket_events(events: list[dict]) -> dict[str, list[str]]:
    buckets = {k: [] for k in EVENT_TYPE_BUCKETS}
    for ev in events:
        etype = ev.get("type", "").lower()
        placed = False
        for bucket, needles in EVENT_TYPE_BUCKETS.items():
            if any(n in etype for n in needles):
                buckets[bucket].append(etype)
                placed = True
                break
        if not placed:
            buckets.setdefault("other", []).append(etype)
    for k in buckets:
        buckets[k] = sorted(set(buckets[k]))
    return buckets


def build_engine_events_payload(
    *,
    case_id: str,
    new_work,
    analysis_success: bool,
    program_path: str,
) -> dict:
    events = events_from_work(new_work)
    return {
        "case_id": case_id,
        "program_path": program_path,
        "analysis_success": analysis_success,
        "derived_license": getattr(new_work, "license_name", None),
        "events": events,
        "engine_events": bucket_events(events),
        "event_count": len(events),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
