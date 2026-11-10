"""Ontology-Struct-F1：自动建模 YAML vs 人工 gold 结构对比。"""
from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from typing import Any, Iterable

from common import BENCHMARK_ROOT, REPO_ROOT, SCRIPTS_DIR
from yml_ontology import load_yml_ontology, resolve_ontology_gold_ref


STRUCT_LIST_FIELDS = (
    "rights",
    "reserved_rights",
    "categories",
    "redistribute",
)

# compat/incompat：SPDX -or-later / -only 与裸 ID 计分等价（如 GPL-2.0-or-later ≈ GPL-2.0）
COMPAT_LIST_FIELDS = ("compat", "incompat")
_COMPAT_ATOM_PREFIXES = ("meta:compat:", "meta:incompat:")
_USER_LICENSE_SUFFIX = "_user_license"
_LICENSE_RAW_DIR = SCRIPTS_DIR / "license_raw"

# 正文点名：法定标题 / 短名。不把「族谱推断」或权利槽（如 commercial_use）当成可验证许可名。
_FAMILY_TITLE_RES: dict[str, tuple[re.Pattern[str], ...]] = {
    "gpl": (
        re.compile(r"gnu general public license", re.IGNORECASE),
        re.compile(r"ordinary gnu general public license", re.IGNORECASE),
        re.compile(r"\bgpl\b", re.IGNORECASE),
    ),
    "agpl": (
        re.compile(r"gnu affero general public license", re.IGNORECASE),
        re.compile(r"affero general public license", re.IGNORECASE),
        re.compile(r"\bagpl\b", re.IGNORECASE),
    ),
    "lgpl": (
        re.compile(r"gnu lesser general public license", re.IGNORECASE),
        re.compile(r"gnu library general public license", re.IGNORECASE),
        re.compile(r"lesser general public license", re.IGNORECASE),
        re.compile(r"library general public license", re.IGNORECASE),
        re.compile(r"\blgpl\b", re.IGNORECASE),
    ),
    "gfdl": (
        re.compile(r"gnu free documentation license", re.IGNORECASE),
        re.compile(r"\bgfdl\b", re.IGNORECASE),
    ),
}

# 元数据层「全体 term 共享」的分发义务（ontology 字段名为 redistribute）
META_GLOBAL_RESTRICTION_FIELDS = ("redistribute", "restrictions")


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _norm_compat_token(value: Any) -> str:
    """SPDX 兼容/不兼容许可 ID：去掉 -or-later / -only 后缀再比较。"""
    tok = _norm_token(value).replace("-", " ")
    for suffix in (" or later", " only"):
        if tok.endswith(suffix):
            tok = tok[: -len(suffix)].strip()
    return " ".join(tok.split())


def _norm_compat_set(items: Iterable[Any] | None) -> set[str]:
    out: set[str] = set()
    for item in items or []:
        tok = _norm_compat_token(item)
        if tok:
            out.add(tok)
    return out


def _is_nodef_result(value: Any) -> bool:
    """result 缺失或 NODEF：许可未明确定义该活动，计分时整行 term 跳过。"""
    return _norm_token(value or "nodef") == "nodef"


def _candidate_license_ids(
    license_id: str | None,
    gold: dict[str, Any] | None,
    pred: dict[str, Any] | None,
) -> list[str]:
    raw: list[str] = []
    for value in (license_id, (gold or {}).get("short_id"), (pred or {}).get("short_id")):
        text = str(value or "").strip()
        if not text:
            continue
        raw.append(text)
        if text.lower().endswith(_USER_LICENSE_SUFFIX):
            raw.append(text[: -len(_USER_LICENSE_SUFFIX)])
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _norm_file_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@lru_cache(maxsize=64)
def _read_license_raw(license_id: str) -> str | None:
    if not license_id or not _LICENSE_RAW_DIR.is_dir():
        return None
    target = _norm_file_key(license_id)
    if not target:
        return None
    for path in sorted(_LICENSE_RAW_DIR.glob("*.txt")):
        if _norm_file_key(path.stem) == target:
            text = path.read_text(encoding="utf-8")
            return text if text.strip() else None
    return None


def resolve_scoring_license_text(
    *,
    license_text: str | None = None,
    license_id: str | None = None,
    gold: dict[str, Any] | None = None,
    pred: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """返回 (正文, 来源)。显式传入的 license_text（含空串）优先于自动加载。"""
    if license_text is not None:
        return license_text, "argument"
    for lid in _candidate_license_ids(license_id, gold, pred):
        text = _read_license_raw(lid)
        if text:
            return text, f"license_raw:{lid}"
    return None, None


def _compat_atom_token(atom: str) -> str | None:
    for prefix in _COMPAT_ATOM_PREFIXES:
        if atom.startswith(prefix):
            return atom[len(prefix) :]
    return None


def _cc_slug(norm: str) -> str | None:
    if not norm.startswith("cc "):
        return None
    rest = norm[3:].strip()
    if not rest.startswith("by"):
        return None
    return rest.replace(" ", "-")


def _named_license_attested(norm_token: str, text: str) -> bool:
    """norm_token 已是 _norm_compat_token（如 'agpl 3.0'、'gpl'、'cc by sa'）。"""
    if not norm_token or not text:
        return False
    compact = norm_token.replace(" ", "-")
    if re.search(rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", text, re.IGNORECASE):
        return True
    cc_slug = _cc_slug(norm_token)
    if cc_slug:
        cc_pats = (
            rf"creativecommons\.org/licenses/{re.escape(cc_slug)}(?:/|\b)",
            rf"(?<![a-z0-9])cc-{re.escape(cc_slug)}(?![a-z0-9])",
            rf"licenses/{re.escape(cc_slug)}/",
        )
        if cc_slug == "by-sa":
            cc_pats += (r"attribution[- ]sharealike",)
        elif cc_slug == "by-nc-sa":
            cc_pats += (r"attribution[- ]noncommercial[- ]sharealike",)
        elif cc_slug == "by-nc":
            cc_pats += (r"attribution[- ]noncommercial(?![- ]sharealike)(?![- ]no?deriv)",)
        elif cc_slug == "by-nd":
            cc_pats += (r"attribution[- ]no[- ]?deriv",)
        elif cc_slug == "by-nc-nd":
            cc_pats += (r"attribution[- ]noncommercial[- ]no[- ]?deriv",)
        return any(re.search(pat, text, re.IGNORECASE) for pat in cc_pats)
    root = norm_token.split()[0]
    for pattern in _FAMILY_TITLE_RES.get(root, ()):
        if pattern.search(text):
            return True
    return False


def _filter_gold_compat_atoms(gold_atoms: set[str], text: str) -> tuple[set[str], list[str]]:
    kept: set[str] = set()
    dropped: list[str] = []
    for atom in gold_atoms:
        token = _compat_atom_token(atom)
        if token is None:
            kept.add(atom)
            continue
        if _named_license_attested(token, text):
            kept.add(atom)
        else:
            dropped.append(atom)
    dropped.sort()
    return kept, dropped


def _norm_set(items: Iterable[Any] | None, *, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for item in items or []:
        tok = _norm_token(item)
        if tok:
            out.add(f"{prefix}{tok}" if prefix else tok)
    return out


def _term_signature(term: dict) -> str:
    usages = ",".join(sorted(_norm_token(u) for u in (term.get("usages") or [])))
    forms = ",".join(sorted(_norm_token(f) for f in (term.get("forms") or [])))
    result = _norm_token(term.get("result") or "nodef")
    restrictions = ",".join(sorted(_norm_token(r) for r in (term.get("restrictions") or [])))
    copyleft = _norm_token(term.get("copyleft"))
    relicense = _norm_token(term.get("relicense"))
    return f"term|{usages}|{forms}|{result}|{restrictions}|{copyleft}|{relicense}"


def _term_merge_key(term: dict) -> tuple[str, str, str]:
    """合并 term 行的键：相同 (result, copyleft, relicense) 视为同行语义。"""
    return (
        _norm_token(term.get("result") or "nodef"),
        _norm_token(term.get("copyleft")),
        _norm_token(term.get("relicense")),
    )


def _dedupe_tokens_by_norm(items: Iterable[Any] | None) -> list[Any]:
    """按归一化值去重，保留首次出现的原始写法，输出按 norm 排序。"""
    seen: dict[str, Any] = {}
    for item in items or []:
        norm = _norm_token(item)
        if norm and norm not in seen:
            seen[norm] = item
    return [seen[k] for k in sorted(seen)]


def normalize_terms_for_compare(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """合并 (result, copyleft, relicense) 相同的 term 行；usages/forms/restrictions 取并集。

    使「拆成多行」与「合并为一行」在计分时等价，不影响 meta 字段。
    """
    if not data or not isinstance(data, dict):
        return data

    d = copy.deepcopy(data)
    terms = d.get("terms") or []
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

    for term in terms:
        if not isinstance(term, dict):
            continue
        key = _term_merge_key(term)
        if key not in buckets:
            buckets[key] = {
                "usage_items": [],
                "form_items": [],
                "restriction_items": [],
                "result": term.get("result"),
                "copyleft": term.get("copyleft"),
                "relicense": term.get("relicense"),
            }
        bucket = buckets[key]
        bucket["usage_items"].extend(term.get("usages") or [])
        bucket["form_items"].extend(term.get("forms") or [])
        bucket["restriction_items"].extend(term.get("restrictions") or [])

    merged_terms: list[dict[str, Any]] = []
    for bucket in buckets.values():
        row: dict[str, Any] = {
            "usages": _dedupe_tokens_by_norm(bucket["usage_items"]),
            "forms": _dedupe_tokens_by_norm(bucket["form_items"]),
            "result": bucket["result"],
            "restrictions": _dedupe_tokens_by_norm(bucket["restriction_items"]),
            "copyleft": bucket["copyleft"],
            "relicense": bucket["relicense"],
        }
        merged_terms.append(row)

    merged_terms.sort(key=_term_signature)
    d["terms"] = merged_terms
    return d


def _meta_global_restriction_items(data: dict[str, Any]) -> list[Any]:
    """元数据 redistribute/restrictions：视为所有非 NODEF term 都应携带的全局义务。"""
    items: list[Any] = []
    for field in META_GLOBAL_RESTRICTION_FIELDS:
        items.extend(data.get(field) or [])
    return items


def prepare_ontology_for_compare(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """term 行合并后，将元数据全局 restrictions 并入各行 restrictions 再计分。"""
    if not data or not isinstance(data, dict):
        return data

    d = normalize_terms_for_compare(data)
    meta_items = _meta_global_restriction_items(d)
    if not meta_items:
        return d

    for term in d.get("terms") or []:
        if not isinstance(term, dict) or _is_nodef_result(term.get("result")):
            continue
        combined = list(term.get("restrictions") or []) + list(meta_items)
        term["restrictions"] = _dedupe_tokens_by_norm(combined)
    return d


def ontology_to_atoms(data: dict[str, Any] | None) -> set[str]:
    """将 ontology dict 展平为可比较的原子集合。"""
    if not data or not isinstance(data, dict):
        return set()

    atoms: set[str] = set()
    for field in STRUCT_LIST_FIELDS:
        for tok in _norm_set(data.get(field)):
            atoms.add(f"meta:{field}:{tok}")

    for field in COMPAT_LIST_FIELDS:
        for tok in _norm_compat_set(data.get(field)):
            atoms.add(f"meta:{field}:{tok}")

    for term in data.get("terms") or []:
        if isinstance(term, dict) and not _is_nodef_result(term.get("result")):
            atoms.add(_term_signature(term))

    return atoms


def _micro_f1(pred: set[str], gold: set[str]) -> dict[str, Any]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(pred),
        "gold_count": len(gold),
        "missing": sorted(gold - pred)[:15],
        "extra": sorted(pred - gold)[:15],
    }


def resolve_case_ontology_gold(case: dict) -> dict[str, Any] | None:
    """从 case.experiment 解析人工 ontology gold。"""
    exp = case.get("experiment") or {}
    rel = exp.get("ontology_gold")
    if rel:
        import yaml

        for base in (BENCHMARK_ROOT, REPO_ROOT):
            path = base / str(rel).replace("/", os.sep)
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data

    ref = exp.get("ontology_gold_ref")
    if ref:
        return resolve_ontology_gold_ref(str(ref))
    return None


def _is_unknown_license_work(work: dict) -> bool:
    return str(work.get("license") or "").lower() in ("unknow", "unknown")


def _placeholder_output_key(work: dict) -> str | None:
    """计分 map 使用的 placeholder key（与 modeling_placeholders 一致）。"""
    if not _is_unknown_license_work(work):
        return None
    ph = work.get("modeling_placeholder")
    if ph:
        return str(ph)
    resolved = work.get("resolved_license")
    if resolved:
        return f"{resolved}_user_license"
    return None


def modeling_placeholders(case: dict) -> list[str]:
    """返回 case 中未知组件的建模 placeholder 名列表。"""
    placeholders: list[str] = []
    for work in (case.get("gold") or {}).get("works") or []:
        key = _placeholder_output_key(work)
        if key:
            placeholders.append(key)
    return placeholders


def registry_lookup_keys_for_unknown_work(work: dict) -> list[str]:
    """按优先级返回 licenses_dict 查找 key（placeholder → resolved → aliases）。"""
    if not _is_unknown_license_work(work):
        return []
    keys: list[str] = []
    for candidate in (
        work.get("modeling_placeholder"),
        work.get("resolved_license"),
        *(work.get("resolved_license_aliases") or []),
    ):
        if candidate:
            text = str(candidate)
            if text not in keys:
                keys.append(text)
    return keys


def capture_from_licenses_dict(
    licenses_dict: dict[str, Any],
    case: dict,
) -> dict[str, dict[str, Any]]:
    """从 licenses_dict 抽取未知组件 LLM 建模 meta（可单测，不依赖 main_case）。"""
    out: dict[str, dict[str, Any]] = {}
    for work in (case.get("gold") or {}).get("works") or []:
        output_key = _placeholder_output_key(work)
        if not output_key:
            continue
        for registry_key in registry_lookup_keys_for_unknown_work(work):
            lic = licenses_dict.get(registry_key)
            if lic is not None and hasattr(lic, "meta"):
                out[output_key] = copy.deepcopy(lic.meta)
                break
    return out


def capture_modeled_ontologies(case: dict) -> dict[str, dict[str, Any]]:
    """exec 后从 par.licenses_dict 抽取 LLM 建模结果。

    Agent codegen 可能写 canonical resolved_license（如 GPL-3.0）而非 placeholder；
    此时仍按 placeholder key 写入 pred map，供 score_case_ontology 计分。
    """
    from common import ensure_scripts_env

    ensure_scripts_env()
    import main_case

    return capture_from_licenses_dict(main_case.par.licenses_dict, case)


def ontology_struct_f1(
    pred: dict[str, Any] | None,
    gold: dict[str, Any] | None,
    *,
    license_text: str | None = None,
    license_id: str | None = None,
) -> dict[str, Any]:
    """单条 ontology 的结构 F1。

    term 按 (result,copyleft,relicense) 合并；short_id/labels/coverage 不计分；
    result 为 NODEF（或缺失）的 term 整行跳过；
    meta.redistribute 仍单独计 meta 原子，并并入各行 term.restrictions 后再比 term。
    gold 的 compat/incompat 仅当对应许可正文点名该许可（法定标题或短名）时才计分；
    预测侧名单仍全计，以便惩罚无原文依据的族谱幻觉。
    无正文时不筛选（保持离线单测）。不再做 proprietary GPL-family incompat 豁免。
    """
    if gold is None:
        return {"skipped": True, "reason": "no_gold_ontology", "f1": None}
    gold_norm = prepare_ontology_for_compare(gold)
    gold_atoms = ontology_to_atoms(gold_norm)
    text, text_source = resolve_scoring_license_text(
        license_text=license_text,
        license_id=license_id,
        gold=gold,
        pred=pred,
    )
    dropped_gold: list[str] = []
    if text is not None:
        gold_atoms, dropped_gold = _filter_gold_compat_atoms(gold_atoms, text)
    if pred is None:
        result = {
            "skipped": False,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(gold_atoms),
        }
    else:
        pred_atoms = ontology_to_atoms(prepare_ontology_for_compare(pred))
        result = _micro_f1(pred_atoms, gold_atoms)
        result["skipped"] = False
    if text is not None:
        result["compat_text_filter"] = {
            "applied": True,
            "source": text_source,
            "dropped_gold": dropped_gold,
        }
    return result


def score_case_ontology(case: dict, pred_ontologies: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    """对 case 中全部未知组件计算 Ontology-Struct-F1。"""
    gold = resolve_case_ontology_gold(case)
    placeholders = modeling_placeholders(case)
    if not placeholders or gold is None:
        return None

    pred_map = pred_ontologies or {}
    details = []
    scores = []
    for ph in placeholders:
        pred = pred_map.get(ph)
        lid = gold.get("short_id") if isinstance(gold, dict) else None
        item = ontology_struct_f1(pred, gold, license_id=str(lid) if lid else None)
        item["placeholder"] = ph
        details.append(item)
        if not item.get("skipped") and item.get("f1") is not None:
            scores.append(float(item["f1"]))

    if not scores:
        return None

    avg = sum(scores) / len(scores)
    return {
        "f1": round(avg, 4),
        "count": len(details),
        "gold_short_id": gold.get("short_id"),
        "details": details,
    }
