"""装配结果的枚举门。合法值与现有 license_data YAML 一致。"""
from __future__ import annotations

from typing import Any

CATEGORIES = frozenset(
    {
        "public",
        "software",
        "data",
        "model",
        "proprietary",
        "permissive",
        "copyleft",
        "disclose",
        "auto-relicensing",
    }
)
RIGHTS = frozenset(
    {
        "use",
        "modify",
        "merge",
        "redistribute",
        "sublicense",
        "commercial_use",
        "patent_use",
        "copyright",
    }
)
RESERVED = frozenset(
    {
        "trademark_use",
        "copyright",
        "patent",
        "trademark",
        "sublicense",
        "commercial_use",
        "patent_use",
        "redistribute",
    }
)
COVERAGE = frozenset({"duplicate", "derivative", "modification", "translation"})
REDISTRIBUTE = frozenset(
    {
        "include_notice",
        "include_license",
        "state_changes",
        "include_use_restriction",
        "include_runtime_restriction",
    }
)
USAGES = frozenset(
    {
        "use",
        "copy",
        "modify",
        "augment",
        "train",
        "combine",
        "combine_mix",
        "amalgamate",
        "distill",
        "generate",
        "embed",
        "stat",
    }
)
FORMS = frozenset({"raw", "binary", "saas"})
RESULTS = frozenset(
    {"duplicate", "independent", "derivative", "modification", "translation", "NODEF"}
)
RELICENSE = (True, False, "conditional")
RESTRICTIONS = REDISTRIBUTE
RIGHTS_PREFIX = frozenset(
    {
        "perpetual",
        "worldwide",
        "non-exclusive",
        "no-charge",
        "royalty-free",
        "irrevocable",
        "non-transferable",
        "non-sublicensable",
        "revocable",
        "sublicensable",
        "any-person",
    }
)
REQUIRED = ("full_name", "short_id", "url", "categories", "rights", "reserved_rights", "terms")


def _list_enum(values: Any, allowed: frozenset, path: str, errors: list[str]) -> None:
    if not isinstance(values, list):
        errors.append(f"{path} 必须是列表")
        return
    for item in values:
        if item not in allowed:
            errors.append(f"{path} 含非法值 {item!r}")


def validate_license_data(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["装配结果不是 dict"]
    for field in REQUIRED:
        if field not in data:
            errors.append(f"缺少必填字段 {field}")
    if "categories" in data:
        _list_enum(data["categories"], CATEGORIES, "categories", errors)
        cats = set(data.get("categories") or [])
        if "proprietary" in cats and "permissive" in cats:
            errors.append("categories 不得同时含 proprietary 与 permissive")
    if "rights" in data:
        _list_enum(data["rights"], RIGHTS, "rights", errors)
    if "reserved_rights" in data:
        _list_enum(data["reserved_rights"], RESERVED, "reserved_rights", errors)
    overlap = set(data.get("rights") or []) & set(data.get("reserved_rights") or [])
    if overlap:
        errors.append(f"rights 与 reserved_rights 交集 {sorted(overlap)}")
    if "coverage" in data:
        _list_enum(data["coverage"], COVERAGE, "coverage", errors)
    if "redistribute" in data:
        _list_enum(data["redistribute"], REDISTRIBUTE, "redistribute", errors)
    if "rights_prefix" in data:
        _list_enum(data["rights_prefix"], RIGHTS_PREFIX, "rights_prefix", errors)
    terms = data.get("terms")
    if "terms" in data and not isinstance(terms, list):
        errors.append("terms 必须是列表")
        return errors
    for i, term in enumerate(terms or []):
        if not isinstance(term, dict):
            errors.append(f"terms[{i}] 不是 dict")
            continue
        prefix = f"terms[{i}]"
        if "usages" not in term or "forms" not in term or "result" not in term:
            errors.append(f"{prefix} 缺少 usages/forms/result")
            continue
        _list_enum(term["usages"], USAGES, f"{prefix}.usages", errors)
        _list_enum(term["forms"], FORMS, f"{prefix}.forms", errors)
        if term["result"] not in RESULTS:
            errors.append(f"{prefix}.result 非法 {term['result']!r}")
        if term["result"] == "NODEF":
            continue
        if "relicense" in term and term["relicense"] not in RELICENSE:
            errors.append(f"{prefix}.relicense 非法 {term['relicense']!r}")
        if "copyleft" in term and term["copyleft"] not in (True, False):
            errors.append(f"{prefix}.copyleft 必须是布尔")
        if "restrictions" in term:
            _list_enum(term["restrictions"], RESTRICTIONS, f"{prefix}.restrictions", errors)
    return errors
