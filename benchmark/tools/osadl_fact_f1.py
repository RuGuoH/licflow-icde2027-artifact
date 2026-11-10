"""OSADL-Fact-F1：在现有 Audit Profile 已投影的事实上计算集合 F1。

不读金标 YAML，不改 osadl_audit_score，也不回写 profile。
金标只含 OSADL 要求出现的原子；约束词表里多出来的原子算误报。
OSADL 没有表态的字段（commercial_use、proprietary、coverage 等）不进词表。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_osadl_modeling import (  # noqa: E402
    MODIFICATION_USAGES,
    COMBINE_USAGES,
    _collect_forms,
    _norm_set,
    _term_has_state_changes,
    _terms_for_usages_and_forms,
    audit_modeling_against_profile,
    load_profile,
    load_yaml,
)
from common import REPO_ROOT  # noqa: E402
from osadl_profile_builder import OSADL_COVERAGE_PROFILES  # noqa: E402

ATOM_COPYLEFT = "categories.copyleft"
ATOM_DISCLOSE = "categories.disclose"
ATOM_NOTICE = "redistribute.include_notice"
ATOM_LICENSE = "redistribute.include_license"
ATOM_SUBLICENSE = "reserved.sublicense"
ATOM_TRADEMARK = "reserved.trademark_use"
ATOM_PATENT = "rights.patent_use"
ATOM_MODIFY_STATE = "modify.state_changes"
ATOM_GLOBAL_STATE = "redistribute.state_changes"
ATOM_SAME_LICENSE = "modify.same_license"
ATOM_MODIFY_COPYLEFT = "modify.copyleft"
ATOM_COMBINE_COPYLEFT = "combine.copyleft"
ATOM_LICENSE_CHOICE = "modify.license_choice"
ATOM_SAAS = "forms.saas"


def _f1(pred: set[str], gold: set[str]) -> dict[str, Any]:
    """与 ontology_metrics._micro_f1 同一公式。两侧皆空时由调用方另判。"""
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
        "missing": sorted(gold - pred),
        "extra": sorted(pred - gold),
    }


def _profile_requires_state_changes(profile: dict[str, Any]) -> bool:
    hints = profile.get("must_not_hints") or {}
    if hints.get("modify_license_notices_ban") or hints.get("modify_legal_notices_ban"):
        return True
    for slice_ in profile.get("terms_slices") or []:
        if "state_changes" in (slice_.get("restrictions") or []):
            return True
    return False


def osadl_fact_inventory(profile: dict[str, Any]) -> tuple[set[str], set[str]]:
    """返回 (gold, vocabulary)。vocabulary 含金标与「明确要求不出现」的原子。"""
    gold: set[str] = set()
    negative: set[str] = set()
    categories = profile.get("categories") or {}
    status = str(profile.get("copyleft_status") or "").lower()
    zero = bool(profile.get("zero_obligations"))
    redistribute = profile.get("redistribute") or {}
    must_not = profile.get("must_not_hints") or {}
    reserved = profile.get("reserved_hints") or {}
    apache = profile.get("apache_hints") or {}
    attribution = profile.get("attribution_hints") or {}
    linking = profile.get("linking_hints") or {}

    if categories.get("copyleft") is True or profile.get("font_delivery"):
        gold.add(ATOM_COPYLEFT)
    elif zero or status in ("no", "questionable") or categories.get("copyleft") is False:
        negative.add(ATOM_COPYLEFT)

    if categories.get("disclose") is True:
        gold.add(ATOM_DISCLOSE)
    elif zero or categories.get("disclose") is False:
        negative.add(ATOM_DISCLOSE)

    if (
        redistribute.get("include_notice")
        or attribution.get("require_credit")
        or apache.get("notice_file")
    ):
        gold.add(ATOM_NOTICE)
    elif zero:
        negative.add(ATOM_NOTICE)

    if redistribute.get("include_license"):
        gold.add(ATOM_LICENSE)
    elif zero:
        negative.add(ATOM_LICENSE)

    if reserved.get("sublicense_prohibited") or must_not.get("modify_license_ban"):
        gold.add(ATOM_SUBLICENSE)

    if (
        must_not.get("promote_ban")
        or must_not.get("trademark_name_ban")
        or apache.get("forward_trademark")
    ):
        gold.add(ATOM_TRADEMARK)

    if apache.get("forward_patent"):
        gold.add(ATOM_PATENT)

    if _profile_requires_state_changes(profile):
        gold.add(ATOM_MODIFY_STATE)
    elif zero:
        negative.add(ATOM_MODIFY_STATE)

    precision = profile.get("redistribute_precision") or {}
    if precision.get("no_global_state_changes"):
        negative.add(ATOM_GLOBAL_STATE)

    if profile.get("grant_same_license"):
        gold.add(ATOM_SAME_LICENSE)

    if profile.get("weak_copyleft") and linking.get("modify_copyleft"):
        gold.add(ATOM_MODIFY_COPYLEFT)
    if profile.get("weak_copyleft") and linking.get("combine_copyleft") is False:
        negative.add(ATOM_COMBINE_COPYLEFT)

    if profile.get("license_choice"):
        gold.add(ATOM_LICENSE_CHOICE)

    disclose_forms = {str(item).lower() for item in (profile.get("disclose_forms") or [])}
    if profile.get("network_service") or "saas" in disclose_forms:
        gold.add(ATOM_SAAS)

    return gold, gold | negative


def _model_satisfies(model: dict[str, Any]) -> set[str]:
    categories = _norm_set(model.get("categories"))
    redistribute = _norm_set(model.get("redistribute"))
    reserved = _norm_set(model.get("reserved_rights"))
    rights = _norm_set(model.get("rights"))
    terms = [t for t in (model.get("terms") or []) if isinstance(t, dict)]
    modify_terms = _terms_for_usages_and_forms(model, usages=MODIFICATION_USAGES)
    combine_terms = _terms_for_usages_and_forms(model, usages=COMBINE_USAGES)

    def has_restriction(key: str) -> bool:
        if key in redistribute:
            return True
        return any(key in _norm_set(term.get("restrictions")) for term in terms)

    satisfied: set[str] = set()
    if "copyleft" in categories:
        satisfied.add(ATOM_COPYLEFT)
    if "disclose" in categories:
        satisfied.add(ATOM_DISCLOSE)
    if has_restriction("include_notice"):
        satisfied.add(ATOM_NOTICE)
    if has_restriction("include_license"):
        satisfied.add(ATOM_LICENSE)
    if "sublicense" in reserved:
        satisfied.add(ATOM_SUBLICENSE)
    if "trademark_use" in reserved:
        satisfied.add(ATOM_TRADEMARK)
    if "patent_use" in rights:
        satisfied.add(ATOM_PATENT)
    modify_state = "state_changes" in redistribute or any(
        _term_has_state_changes(term) for term in modify_terms
    )
    if modify_state:
        satisfied.add(ATOM_MODIFY_STATE)
    if "state_changes" in redistribute:
        satisfied.add(ATOM_GLOBAL_STATE)
    if any(term.get("relicense") is False or term.get("copyleft") is True for term in modify_terms):
        satisfied.add(ATOM_SAME_LICENSE)
    if any(term.get("copyleft") is True for term in modify_terms):
        satisfied.add(ATOM_MODIFY_COPYLEFT)
    if any(term.get("copyleft") is True for term in combine_terms):
        satisfied.add(ATOM_COMBINE_COPYLEFT)
    if any(term.get("relicense") in (False, "conditional") for term in modify_terms):
        satisfied.add(ATOM_LICENSE_CHOICE)
    if "saas" in _collect_forms(model):
        satisfied.add(ATOM_SAAS)
    return satisfied


def osadl_fact_f1(model: dict[str, Any] | None, profile: dict[str, Any]) -> dict[str, Any]:
    """单份许可的 OSADL-Fact-F1。model 为空集时按空建模计。"""
    gold, vocabulary = osadl_fact_inventory(profile)
    satisfied = set() if model is None else _model_satisfies(model)
    pred = satisfied & vocabulary
    if not gold and not pred:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "pred_count": 0,
            "gold_count": 0,
            "missing": [],
            "extra": [],
            "vacuous": True,
            "gold": [],
            "pred": [],
        }
    result = _f1(pred, gold)
    result["vacuous"] = False
    result["gold"] = sorted(gold)
    result["pred"] = sorted(pred)
    return result


def _find_pred(license_id: str, pred_dirs: list[Path]) -> Path | None:
    for pred_dir in pred_dirs:
        path = pred_dir / license_id / "license_final.yaml"
        if path.is_file():
            return path
    return None


def score_osadl_fact_f1(
    pred_dirs: list[Path],
    *,
    license_ids: list[str] | None = None,
) -> dict[str, Any]:
    """对 coverage 许可计 OSADL-Fact-F1，并附带未改动的扣分审计结果。"""
    ids = list(license_ids or OSADL_COVERAGE_PROFILES)
    rows: list[dict[str, Any]] = []
    micro_gold: set[str] = set()
    micro_pred: set[str] = set()
    for lic_id in ids:
        profile = load_profile(lic_id)
        pred_path = _find_pred(lic_id, pred_dirs)
        row: dict[str, Any] = {"license_id": lic_id}
        if profile is None or pred_path is None:
            row["skipped"] = True
            row["reason"] = "no_osadl_profile" if profile is None else "no_pred"
            rows.append(row)
            continue
        model = load_yaml(pred_path)
        fact = osadl_fact_f1(model, profile)
        audit = audit_modeling_against_profile(model, profile)
        row.update(fact)
        row["skipped"] = False
        row["pred_path"] = str(pred_path.relative_to(REPO_ROOT)).replace("\\", "/")
        row["osadl_audit_score"] = audit["osadl_audit_score"]
        row["usable_for_compliance"] = audit["usable_for_compliance"]
        if not fact["vacuous"]:
            micro_gold.update(f"{lic_id}|{atom}" for atom in fact["gold"])
            micro_pred.update(f"{lic_id}|{atom}" for atom in fact["pred"])
        rows.append(row)

    scored = [row for row in rows if not row.get("skipped")]
    mean_f1 = round(sum(row["f1"] for row in scored) / len(scored), 4) if scored else None
    micro = _f1(micro_pred, micro_gold)
    mean_audit = (
        round(sum(row["osadl_audit_score"] for row in scored) / len(scored), 4) if scored else None
    )
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "pred_dirs": [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in pred_dirs],
        "rows": rows,
        "summary": {
            "scored": len(scored),
            "skipped": sum(1 for row in rows if row.get("skipped")),
            "vacuous": sum(1 for row in scored if row.get("vacuous")),
            "mean_f1": mean_f1,
            "micro_f1": micro["f1"],
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_tp": micro["tp"],
            "micro_fp": micro["fp"],
            "micro_fn": micro["fn"],
            "mean_osadl_audit_score": mean_audit,
            "usable_count": sum(1 for row in scored if row.get("usable_for_compliance")),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score modeling YAML with OSADL-Fact-F1")
    parser.add_argument("--pred-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--license", action="append", dest="licenses")
    args = parser.parse_args(argv)

    pred_dirs = [REPO_ROOT / path for path in args.pred_dir]
    for pred_dir in pred_dirs:
        if not pred_dir.is_dir():
            print(f"ERROR: pred dir not found: {pred_dir}", file=sys.stderr)
            return 1
    report = score_osadl_fact_f1(pred_dirs, license_ids=args.licenses)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"=== OSADL-Fact-F1 -> {out_path} ===")
    print(
        f"  scored={summary['scored']}  mean_f1={summary['mean_f1']}  "
        f"micro_f1={summary['micro_f1']}  "
        f"audit_mean={summary['mean_osadl_audit_score']}  "
        f"usable={summary['usable_count']}/{summary['scored']}"
    )
    for row in report["rows"]:
        if row.get("skipped"):
            print(f"  [{row['license_id']}] skipped: {row.get('reason')}")
            continue
        print(
            f"  [{row['license_id']}] f1={row['f1']} p={row['precision']} r={row['recall']} "
            f"audit={row['osadl_audit_score']}"
        )
        if row.get("missing"):
            print(f"    missing: {', '.join(row['missing'])}")
        if row.get("extra"):
            print(f"    extra: {', '.join(row['extra'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
