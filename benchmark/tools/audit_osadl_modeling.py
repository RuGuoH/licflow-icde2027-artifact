"""Audit ModelGo license modeling YAML against OSADL audit profiles."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from common import REPO_ROOT  # noqa: E402
from osadl_profile_builder import OSADL_COVERAGE_PROFILES  # noqa: E402

PROFILE_DIR = REPO_ROOT / "scripts" / "osadl_unreflicenses" / "profiles"

MODIFICATION_USAGES = frozenset({"modify", "amalgamate", "merge"})
COMBINE_USAGES = frozenset({"combine", "combine_mix"})
NOTICE_KEYS = frozenset({"include_notice"})
LICENSE_KEYS = frozenset({"include_license"})


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_profile(license_id: str, *, profile_dir: Path = PROFILE_DIR) -> dict[str, Any] | None:
    path = profile_dir / f"{license_id}.yaml"
    if not path.is_file():
        return None
    return load_yaml(path)


def _norm_set(items: Any) -> set[str]:
    if not items:
        return set()
    return {str(x).strip().lower() for x in items}


def _issue(
    *,
    severity: str,
    field: str,
    kind: str,
    expected: Any,
    actual: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "field": field,
        "kind": kind,
        "expected": expected,
        "actual": actual,
        "message": message,
    }


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for issue in issues:
        key = (issue["severity"], issue["field"], issue["message"])
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _term_satisfies_source_disclosure(
    term: dict[str, Any],
    model: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    """源码披露义务代理：与 term.copyleft / 弱 copyleft 规则分离。"""
    if term.get("copyleft") is True:
        return True
    restr = _norm_set(term.get("restrictions"))
    if restr & (NOTICE_KEYS | LICENSE_KEYS | frozenset({"state_changes"})):
        return True
    if term.get("relicense") in (False, "conditional"):
        return True
    prof_cat = profile.get("categories") or {}
    if prof_cat.get("disclose") and "disclose" in _norm_set(model.get("categories")):
        return True
    return False


def _terms_matching_slice(model: dict[str, Any], slice_: dict[str, Any]) -> list[dict[str, Any]]:
    want_forms = set(slice_.get("forms") or [])
    want_when = set(slice_.get("when") or [])
    out: list[dict[str, Any]] = []
    for term in model.get("terms") or []:
        if not isinstance(term, dict):
            continue
        forms = set(term.get("forms") or [])
        if want_forms and not (forms & want_forms):
            continue
        usages = _norm_set(term.get("usages"))
        if "software_modification" in want_when and not (usages & MODIFICATION_USAGES):
            continue
        out.append(term)
    return out


def _term_has_state_changes(term: dict[str, Any]) -> bool:
    return "state_changes" in _norm_set(term.get("restrictions"))


def _collect_forms(model: dict[str, Any]) -> set[str]:
    forms: set[str] = set()
    for term in model.get("terms") or []:
        if isinstance(term, dict):
            forms.update(_norm_set(term.get("forms")))
    return forms


def _terms_for_usages_and_forms(
    model: dict[str, Any],
    *,
    usages: frozenset[str],
    forms: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in model.get("terms") or []:
        if not isinstance(term, dict):
            continue
        if not (_norm_set(term.get("usages")) & usages):
            continue
        if forms is not None and not (_norm_set(term.get("forms")) & forms):
            continue
        out.append(term)
    return out


def audit_modeling_against_profile(
    model: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    license_id = profile.get("short_id") or model.get("short_id") or "unknown"
    provides = set(profile.get("_audit", {}).get("provides") or [])
    issues: list[dict[str, Any]] = []

    categories = _norm_set(model.get("categories"))
    redistribute = _norm_set(model.get("redistribute"))
    reserved = _norm_set(model.get("reserved_rights"))
    rights = _norm_set(model.get("rights"))
    terms = [t for t in (model.get("terms") or []) if isinstance(t, dict)]

    if "copyleft_status" in provides:
        status = str(profile.get("copyleft_status") or "").lower()
        if status == "questionable" and "copyleft" in categories:
            issues.append(
                _issue(
                    severity="warning",
                    field="categories.copyleft",
                    kind="precision",
                    expected="omit copyleft (OSADL questionable)",
                    actual=True,
                    message="OSADL copyleft 为 questionable，建模不宜断言 categories.copyleft",
                )
            )
        elif status == "no" and "copyleft" in categories:
            issues.append(
                _issue(
                    severity="critical",
                    field="categories.copyleft",
                    kind="precision",
                    expected=False,
                    actual=True,
                    message="OSADL 无 copyleft 信号，建模不应标注 copyleft",
                )
            )

    if profile.get("zero_obligations"):
        if redistribute:
            issues.append(
                _issue(
                    severity="warning",
                    field="redistribute",
                    kind="precision",
                    expected=[],
                    actual=sorted(redistribute),
                    message="OSADL 零义务许可，redistribute 宜为空",
                )
            )
        for key in ("copyleft", "disclose"):
            if key in categories:
                issues.append(
                    _issue(
                        severity="warning",
                        field=f"categories.{key}",
                        kind="precision",
                        expected=False,
                        actual=True,
                        message=f"OSADL 零义务许可，不宜标注 categories.{key}",
                    )
                )

    if "categories" in provides:
        prof_cat = profile.get("categories") or {}
        for key in ("copyleft", "disclose"):
            if key not in prof_cat:
                continue
            expected = bool(prof_cat[key])
            actual = key in categories
            if expected and not actual:
                issues.append(
                    _issue(
                        severity="critical",
                        field=f"categories.{key}",
                        kind="recall",
                        expected=True,
                        actual=False,
                        message=f"OSADL 要求 categories 含 {key}，建模缺失",
                    )
                )
            elif not expected and actual:
                issues.append(
                    _issue(
                        severity="critical",
                        field=f"categories.{key}",
                        kind="precision",
                        expected=False,
                        actual=True,
                        message=f"OSADL 无 {key} 信号，建模不应标注 {key}",
                    )
                )

    if "disclose_forms" in provides:
        expected_forms = [str(f).lower() for f in profile.get("disclose_forms") or []]
        if "saas" in expected_forms and "saas" not in _collect_forms(model):
            issues.append(
                _issue(
                    severity="warning",
                    field="terms.forms",
                    kind="recall",
                    expected="saas",
                    actual=sorted(_collect_forms(model)),
                    message="OSADL 在网络/ saas 场景有披露义务，建模 terms 未出现 saas",
                )
            )

    if "redistribute" in provides:
        prof_red = profile.get("redistribute") or {}
        for key, expected in prof_red.items():
            if not expected:
                continue
            has_meta = key in redistribute
            has_term = any(key in _norm_set(t.get("restrictions")) for t in terms)
            if not (has_meta or has_term):
                issues.append(
                    _issue(
                        severity="warning" if key in NOTICE_KEYS | LICENSE_KEYS else "critical",
                        field=f"redistribute.{key}",
                        kind="recall",
                        expected=True,
                        actual=False,
                        message=f"OSADL 要求 {key}，建模 redistribute/terms 均未体现",
                    )
                )

    needs_sublicense_reserved = False
    if "reserved_hints" in provides:
        hints = profile.get("reserved_hints") or {}
        if hints.get("sublicense_prohibited"):
            needs_sublicense_reserved = True
    if "must_not_hints" in provides:
        hints = profile.get("must_not_hints") or {}
        if hints.get("modify_license_ban"):
            needs_sublicense_reserved = True
    if needs_sublicense_reserved and "sublicense" not in reserved:
        issues.append(
            _issue(
                severity="warning",
                field="reserved_rights.sublicense",
                kind="recall",
                expected=True,
                actual=False,
                message="OSADL 禁止再许可/修改许可证，reserved_rights 宜含 sublicense",
            )
        )

    if profile.get("grant_same_license"):
        candidates = [
            t
            for t in terms
            if _norm_set(t.get("usages")) & MODIFICATION_USAGES
            and _norm_set(t.get("forms")) & {"raw", "binary"}
        ]
        ok = any(
            t.get("relicense") is False or t.get("copyleft") is True for t in candidates
        )
        if candidates and not ok:
            issues.append(
                _issue(
                    severity="critical",
                    field="terms.relicense",
                    kind="recall",
                    expected="relicense:false or copyleft:true on modify/amalgamate",
                    actual=[
                        {
                            "usages": t.get("usages"),
                            "forms": t.get("forms"),
                            "relicense": t.get("relicense"),
                            "copyleft": t.get("copyleft"),
                        }
                        for t in candidates
                    ],
                    message="OSADL 要求修改后同许可再授权，modify/amalgamate 应为 relicense:false 或 copyleft:true",
                )
            )

    if "must_not_hints" in provides:
        hints = profile.get("must_not_hints") or {}
        if hints.get("promote_ban") and "trademark_use" not in reserved:
            issues.append(
                _issue(
                    severity="warning",
                    field="reserved_rights.trademark_use",
                    kind="recall",
                    expected=True,
                    actual=False,
                    message="OSADL 禁止 Promote/背书，reserved_rights 宜含 trademark_use",
                )
            )
        if hints.get("trademark_name_ban") and "trademark_use" not in reserved:
            issues.append(
                _issue(
                    severity="warning",
                    field="reserved_rights.trademark_use",
                    kind="recall",
                    expected=True,
                    actual=False,
                    message="OSADL 限制商标/产品名使用，reserved_rights 宜含 trademark_use",
                )
            )
        if hints.get("modify_license_notices_ban") or hints.get("modify_legal_notices_ban"):
            modify_terms = _terms_for_usages_and_forms(
                model, usages=MODIFICATION_USAGES, forms={"raw", "binary"}
            )
            if modify_terms and not any(_term_has_state_changes(t) for t in modify_terms):
                if "state_changes" not in redistribute:
                    issues.append(
                        _issue(
                            severity="warning",
                            field="terms.restrictions",
                            kind="recall",
                            expected="state_changes",
                            actual=[t.get("restrictions") for t in modify_terms],
                            message="OSADL 禁止改许可证/法律声明，modify 场景宜标注 state_changes",
                        )
                    )

    if "apache_hints" in provides:
        hints = profile.get("apache_hints") or {}
        if hints.get("forward_patent") and "patent_use" not in rights:
            issues.append(
                _issue(
                    severity="warning",
                    field="rights.patent_use",
                    kind="recall",
                    expected=True,
                    actual=False,
                    message="OSADL 要求转发专利许可，rights 宜含 patent_use",
                )
            )
        if hints.get("forward_trademark") and "trademark_use" not in reserved:
            issues.append(
                _issue(
                    severity="warning",
                    field="reserved_rights.trademark_use",
                    kind="recall",
                    expected=True,
                    actual=False,
                    message="OSADL 转发商标限制，reserved_rights 宜含 trademark_use",
                )
            )
        if hints.get("notice_file"):
            modify_raw = _terms_for_usages_and_forms(
                model, usages=MODIFICATION_USAGES, forms={"raw"}
            )
            has_notice = "include_notice" in redistribute or any(
                "include_notice" in _norm_set(t.get("restrictions")) for t in modify_raw
            )
            if modify_raw and not has_notice:
                issues.append(
                    _issue(
                        severity="warning",
                        field="terms.restrictions",
                        kind="recall",
                        expected="include_notice",
                        actual=[t.get("restrictions") for t in modify_raw],
                        message="OSADL 含 NOTICE 文件义务，raw 修改场景宜含 include_notice",
                    )
                )

    if "redistribute_precision" in provides:
        precision = profile.get("redistribute_precision") or {}
        if precision.get("no_global_state_changes") and "state_changes" in redistribute:
            issues.append(
                _issue(
                    severity="warning",
                    field="redistribute.state_changes",
                    kind="precision",
                    expected=False,
                    actual=True,
                    message="OSADL 仅在修改场景要求 state_changes，不宜放在全局 redistribute",
                )
            )

    if "redistribute_slices" in provides:
        for slice_ in profile.get("redistribute_slices") or []:
            want_forms = set(slice_.get("forms") or [])
            for key in ("include_notice", "include_license"):
                if not slice_.get(key):
                    continue
                matched = _terms_for_usages_and_forms(
                    model,
                    usages=frozenset({"copy", "use", "share", "sell", "modify", "combine"}),
                    forms=want_forms,
                )
                has_meta = key in redistribute
                has_term = any(key in _norm_set(t.get("restrictions")) for t in matched)
                if not (has_meta or has_term):
                    issues.append(
                        _issue(
                            severity="warning",
                            field=f"redistribute_slices.{key}",
                            kind="recall",
                            expected={"forms": sorted(want_forms), key: True},
                            actual=sorted(_collect_forms(model)),
                            message=f"OSADL 在 forms={sorted(want_forms)} 要求 {key}",
                        )
                    )
            want_notice = slice_.get("include_notice")
            want_license = slice_.get("include_license")
            if want_notice is False or want_license is False:
                continue
            if want_notice and not want_license:
                for term in _terms_for_usages_and_forms(
                    model, usages=frozenset({"copy"}), forms=want_forms
                ):
                    extra = _norm_set(term.get("restrictions")) & NOTICE_KEYS
                    if extra and "include_license" not in _norm_set(term.get("restrictions")):
                        continue
                    if NOTICE_KEYS <= _norm_set(term.get("restrictions")) and "raw" not in (
                        _norm_set(term.get("forms")) & want_forms
                    ):
                        issues.append(
                            _issue(
                                severity="warning",
                                field="terms.restrictions",
                                kind="precision",
                                expected="license without notice on binary-only copy",
                                actual=term.get("restrictions"),
                                message="OSADL binary 场景可能仅需 license，copy term 不宜多标 notice",
                            )
                        )

    if profile.get("weak_copyleft") and profile.get("linking_hints"):
        hints = profile["linking_hints"]
        modify_terms = _terms_for_usages_and_forms(
            model, usages=MODIFICATION_USAGES, forms={"raw", "binary"}
        )
        combine_terms = _terms_for_usages_and_forms(
            model, usages=COMBINE_USAGES, forms={"raw", "binary"}
        )
        if hints.get("modify_copyleft") and modify_terms:
            if not any(t.get("copyleft") is True for t in modify_terms):
                issues.append(
                    _issue(
                        severity="critical",
                        field="terms.copyleft",
                        kind="recall",
                        expected=True,
                        actual=[t.get("copyleft") for t in modify_terms],
                        message="弱 copyleft：modify/amalgamate 场景宜 copyleft:true",
                    )
                )
        if hints.get("combine_copyleft") is False and combine_terms:
            if any(t.get("copyleft") is True for t in combine_terms):
                issues.append(
                    _issue(
                        severity="critical",
                        field="terms.copyleft",
                        kind="precision",
                        expected=False,
                        actual=[t.get("copyleft") for t in combine_terms],
                        message="弱 copyleft：combine 链接场景不宜 copyleft:true",
                    )
                )

    if "attribution_hints" in provides:
        hints = profile.get("attribution_hints") or {}
        if hints.get("require_credit"):
            has_notice = "include_notice" in redistribute or any(
                "include_notice" in _norm_set(t.get("restrictions")) for t in terms
            )
            if not has_notice:
                issues.append(
                    _issue(
                        severity="warning",
                        field="redistribute.include_notice",
                        kind="recall",
                        expected=True,
                        actual=False,
                        message="OSADL 要求 credit/署名，宜含 include_notice",
                    )
                )

    if profile.get("license_choice"):
        mod_terms = _terms_for_usages_and_forms(
            model, usages=MODIFICATION_USAGES, forms={"raw", "binary"}
        )
        if mod_terms and not any(t.get("relicense") in (False, "conditional") for t in mod_terms):
            issues.append(
                _issue(
                    severity="warning",
                    field="terms.relicense",
                    kind="recall",
                    expected="conditional or false",
                    actual=[t.get("relicense") for t in mod_terms],
                    message="OSADL 要求 compatible/original license 选择，modify 宜 relicense:conditional",
                )
            )

    if profile.get("font_delivery") and "copyleft" not in categories:
        issues.append(
            _issue(
                severity="warning",
                field="categories.copyleft",
                kind="recall",
                expected=True,
                actual=False,
                message="OFL 为 copyleft 字体许可，categories 宜含 copyleft",
            )
        )

    apache_hints = profile.get("apache_hints") or {}
    if apache_hints.get("forward_patent") and "patent_use" not in rights:
        issues.append(
            _issue(
                severity="warning",
                field="rights.patent_use",
                kind="recall",
                expected=True,
                actual=False,
                message="OSADL 要求转发专利声明，rights 宜含 patent_use",
            )
        )

    if "terms_slices" in provides:
        for slice_ in profile.get("terms_slices") or []:
            matched = _terms_matching_slice(model, slice_)
            restrictions = slice_.get("restrictions") or []
            obligations = slice_.get("obligations") or []

            if "state_changes" in restrictions:
                if not matched:
                    issues.append(
                        _issue(
                            severity="warning",
                            field="terms",
                            kind="recall",
                            expected=slice_,
                            actual=[],
                            message="OSADL 要求 modification 场景 state_changes，但无匹配 term",
                        )
                    )
                elif not any(_term_has_state_changes(t) for t in matched):
                    issues.append(
                        _issue(
                            severity="critical",
                            field="terms.restrictions",
                            kind="recall",
                            expected="state_changes",
                            actual=[t.get("restrictions") for t in matched],
                            message=f"OSADL 在 forms={slice_.get('forms')} 修改场景要求 state_changes",
                        )
                    )

            if "source_disclosure" in obligations:
                mod_matched = [
                    t for t in matched if _norm_set(t.get("usages")) & MODIFICATION_USAGES
                ]
                if not matched:
                    issues.append(
                        _issue(
                            severity="warning",
                            field="terms",
                            kind="recall",
                            expected=slice_,
                            actual=[],
                            message="OSADL 要求修改场景源码披露，但无匹配 term",
                        )
                    )
                elif not mod_matched:
                    issues.append(
                        _issue(
                            severity="warning",
                            field="terms",
                            kind="recall",
                            expected=slice_,
                            actual=[t.get("usages") for t in matched],
                            message="OSADL 要求修改场景源码披露，但无 modify/amalgamate term",
                        )
                    )
                elif not any(
                    _term_satisfies_source_disclosure(t, model, profile) for t in mod_matched
                ):
                    issues.append(
                        _issue(
                            severity="warning",
                            field="terms.source_disclosure",
                            kind="recall",
                            expected="notice/license/state_changes, relicense:false|conditional, or categories.disclose",
                            actual=[
                                {
                                    "usages": t.get("usages"),
                                    "forms": t.get("forms"),
                                    "restrictions": t.get("restrictions"),
                                    "relicense": t.get("relicense"),
                                    "copyleft": t.get("copyleft"),
                                }
                                for t in mod_matched
                            ],
                            message="修改场景源码披露宜在 modify term 体现（restrictions/relicense/disclose，非 term.copyleft）",
                        )
                    )

            # precision: profile slice only raw, but MG puts state_changes on binary too
            if (
                "state_changes" in restrictions
                and slice_.get("forms") == ["raw"]
                and "software_modification" in (slice_.get("when") or [])
            ):
                for term in terms:
                    usages = _norm_set(term.get("usages")) & MODIFICATION_USAGES
                    forms = _norm_set(term.get("forms"))
                    if not usages or "raw" in forms:
                        continue
                    if _term_has_state_changes(term):
                        issues.append(
                            _issue(
                                severity="warning",
                                field="terms.restrictions",
                                kind="precision",
                                expected="no state_changes on binary/saas-only modify",
                                actual={
                                    "usages": term.get("usages"),
                                    "forms": term.get("forms"),
                                    "restrictions": term.get("restrictions"),
                                },
                                message="OSADL 仅在 raw 修改场景要求 state_changes，binary/saas 不应多标",
                            )
                        )

    if profile.get("network_service") and "saas" not in _collect_forms(model):
        issues.append(
            _issue(
                severity="warning",
                field="terms.forms",
                kind="recall",
                expected="saas",
                actual=sorted(_collect_forms(model)),
                message="OSADL 含 Network service，建模宜包含 saas form",
            )
        )

    issues = _dedupe_issues(issues)
    critical = sum(1 for i in issues if i["severity"] == "critical")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    score = max(0.0, 1.0 - 0.2 * critical - 0.05 * warnings)

    return {
        "license_id": license_id,
        "profile_version": profile.get("_audit", {}).get("profile_version"),
        "osadl_file": profile.get("_audit", {}).get("osadl_file"),
        "schema_gaps": profile.get("_audit", {}).get("schema_gaps") or [],
        "usable_for_compliance": critical == 0,
        "osadl_audit_score": round(score, 4),
        "issue_counts": {"critical": critical, "warning": warnings, "total": len(issues)},
        "issues": issues,
    }


def audit_pred_dir(
    pred_dir: Path,
    *,
    license_ids: list[str] | None = None,
    profile_dir: Path = PROFILE_DIR,
) -> dict[str, Any]:
    ids = list(license_ids or OSADL_COVERAGE_PROFILES)
    rows: list[dict[str, Any]] = []
    for lic_id in ids:
        profile = load_profile(lic_id, profile_dir=profile_dir)
        pred_path = pred_dir / lic_id / "license_final.yaml"
        row: dict[str, Any] = {"license_id": lic_id, "pred_path": str(pred_path)}
        if profile is None:
            row["skipped"] = True
            row["reason"] = "no_osadl_profile"
            rows.append(row)
            continue
        if not pred_path.is_file():
            row["skipped"] = True
            row["reason"] = "no_pred"
            rows.append(row)
            continue
        model = load_yaml(pred_path)
        row.update(audit_modeling_against_profile(model, profile))
        row["skipped"] = False
        rows.append(row)

    scored = [r for r in rows if not r.get("skipped")]
    mean_score = (
        round(sum(r["osadl_audit_score"] for r in scored) / len(scored), 4) if scored else None
    )
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "pred_dir": str(pred_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "profile_dir": str(profile_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rows": rows,
        "summary": {
            "audited": len(scored),
            "skipped": sum(1 for r in rows if r.get("skipped")),
            "mean_osadl_audit_score": mean_score,
            "usable_count": sum(1 for r in scored if r.get("usable_for_compliance")),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit license modeling against OSADL profiles")
    parser.add_argument(
        "--pred-dir",
        default="benchmark/results/rq4_modeling_v5_flash",
        help="batch_license_modeling 输出目录",
    )
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    parser.add_argument("--out", help="JSON 报告路径")
    parser.add_argument("--license", action="append", dest="licenses")
    args = parser.parse_args(argv)

    pred_dir = REPO_ROOT / args.pred_dir
    if not pred_dir.is_dir():
        print(f"ERROR: pred dir not found: {pred_dir}", file=sys.stderr)
        return 1

    report = audit_pred_dir(
        pred_dir,
        license_ids=args.licenses,
        profile_dir=args.profile_dir,
    )
    out_path = (
        Path(args.out)
        if args.out
        else pred_dir / "osadl_audit_report.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(f"=== OSADL audit -> {out_path} ===")
    print(
        f"  audited={summary['audited']}  mean_score={summary['mean_osadl_audit_score']}  "
        f"usable={summary['usable_count']}/{summary['audited']}"
    )
    for row in report["rows"]:
        if row.get("skipped"):
            print(f"  [{row['license_id']}] skipped: {row.get('reason')}")
            continue
        print(
            f"  [{row['license_id']}] score={row['osadl_audit_score']} "
            f"critical={row['issue_counts']['critical']} warning={row['issue_counts']['warning']}"
        )
        for issue in row.get("issues") or []:
            print(f"    - {issue['severity']}: {issue['message']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
