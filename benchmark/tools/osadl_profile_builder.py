"""Build ModelGo-like audit profiles from OSADL optimized JSON checklists."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common import REPO_ROOT

OSADL_ROOT = REPO_ROOT / "scripts" / "osadl_unreflicenses"
JSON_OPT_DIR = OSADL_ROOT / "json-opt"
TXT_DIR = OSADL_ROOT
DEFAULT_PROFILE_DIR = OSADL_ROOT / "profiles"

# Curated non-exhaustive OSADL obligation-family coverage (20 licenses / 27 clusters).
OSADL_COVERAGE_PROFILES: tuple[str, ...] = (
    # permissive / patent
    "MIT",
    "Apache-2.0",
    "0BSD",
    "BSD-3-Clause",
    "BSD-2-Clause-Patent",
    # strong / network copyleft
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    # weak copyleft / alternate families
    "MPL-2.0",
    "MPL-2.0-no-copyleft-exception",
    "EPL-2.0",
    "OSL-3.0",
    # disclosure / edge families
    "AFL-3.0",
    "Artistic-2.0",
    "OFL-1.1",
    "OpenSSL",
    "MS-PL",
    "MS-RL",
)

# Not in OSADL_COVERAGE_PROFILES: OSADL json-opt exists but MG modeling/audit is out of scope.
OSADL_COVERAGE_EXCLUDED: tuple[tuple[str, str], ...] = (
    (
        "GPL-2.0-classpath",
        "license full text unavailable (only GPL-2.0 body; classpath exception text missing)",
    ),
)

# ModelGo short_id -> OSADL json-opt filename (without path)
MG_TO_OSADL_FILE: dict[str, str] = {
    "MIT": "MIT-opt.json",
    "Apache-2.0": "Apache-2.0-opt.json",
    "0BSD": "0BSD-opt.json",
    "BSD-3-Clause": "BSD-3-Clause-opt.json",
    "BSD-2-Clause-Patent": "BSD-2-Clause-Patent-opt.json",
    "GPL-2.0": "GPL-2.0-only-opt.json",
    "GPL-2.0-only": "GPL-2.0-only-opt.json",
    "GPL-2.0-or-later": "GPL-2.0-or-later-opt.json",
    "GPL-2.0-classpath": "GPL-2.0-only WITH Classpath-exception-2.0-opt.json",
    "GPL-3.0": "GPL-3.0-only-opt.json",
    "GPL-3.0-only": "GPL-3.0-only-opt.json",
    "GPL-3.0-or-later": "GPL-3.0-or-later-opt.json",
    "AGPL-3.0": "AGPL-3.0-only-opt.json",
    "AGPL-3.0-only": "AGPL-3.0-only-opt.json",
    "AGPL-3.0-or-later": "AGPL-3.0-or-later-opt.json",
    "LGPL-2.1": "LGPL-2.1-only-opt.json",
    "LGPL-2.1-only": "LGPL-2.1-only-opt.json",
    "LGPL-2.1-or-later": "LGPL-2.1-or-later-opt.json",
    "LGPL-3.0": "LGPL-3.0-only-opt.json",
    "LGPL-3.0-only": "LGPL-3.0-only-opt.json",
    "LGPL-3.0-or-later": "LGPL-3.0-or-later-opt.json",
    "MPL-2.0": "MPL-2.0-opt.json",
    "MPL-2.0-no-copyleft-exception": "MPL-2.0-no-copyleft-exception-opt.json",
    "EPL-2.0": "EPL-2.0-opt.json",
    "OSL-3.0": "OSL-3.0-opt.json",
    "AFL-3.0": "AFL-3.0-opt.json",
    "Artistic-2.0": "Artistic-2.0-opt.json",
    "OFL-1.1": "OFL-1.1-opt.json",
    "OpenSSL": "OpenSSL-opt.json",
    "MS-PL": "MS-PL-opt.json",
    "MS-RL": "MS-RL-opt.json",
}

USE_CASE_FORMS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"network service", re.I), ["saas"]),
    (re.compile(r"source code delivery", re.I), ["raw"]),
    (re.compile(r"binary delivery", re.I), ["binary"]),
]

META_KEYS = frozenset(
    {
        "COMPATIBILITY",
        "INCOMPATIBILITY",
        "DEPENDING COMPATIBILITY",
        "COPYLEFT CLAUSE",
        "PATENT HINTS",
    }
)
BRANCH_KEYS = frozenset({"EITHER", "OR", "EITHER IF", "OR IF"})


@dataclass
class WalkContext:
    use_case: str | None = None
    forms: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    def with_use_case(self, name: str) -> WalkContext:
        forms = tuple(_forms_from_use_case(name))
        return WalkContext(use_case=name, forms=forms, conditions=self.conditions)

    def with_condition(self, cond: str) -> WalkContext:
        return WalkContext(
            use_case=self.use_case,
            forms=self.forms,
            conditions=self.conditions + (cond,),
        )


@dataclass
class ObligationRecord:
    polarity: str  # must | must_not
    action: str
    ctx: WalkContext


@dataclass
class ProfileAccumulator:
    license_id: str
    osadl_file: str
    copyleft: str | None = None  # yes | no | questionable
    patent_hints: bool | None = None
    obligations: list[ObligationRecord] = field(default_factory=list)
    grant_license_actions: list[ObligationRecord] = field(default_factory=list)

    def add_must(self, action: str, ctx: WalkContext) -> None:
        rec = ObligationRecord("must", _norm_action(action), ctx)
        self.obligations.append(rec)
        if action.lower().startswith("grant license"):
            self.grant_license_actions.append(rec)

    def add_must_not(self, action: str, ctx: WalkContext) -> None:
        self.obligations.append(ObligationRecord("must_not", _norm_action(action), ctx))


def _norm_action(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _forms_from_use_case(name: str) -> list[str]:
    forms: list[str] = []
    for pattern, labels in USE_CASE_FORMS:
        if pattern.search(name):
            for label in labels:
                if label not in forms:
                    forms.append(label)
    return forms


def _is_notice_action(action: str) -> bool:
    a = action.lower()
    return "copyright" in a and any(
        v in a for v in ("provide", "forward", "display", "credit")
    )


def _is_license_text_action(action: str) -> bool:
    """Full license text / LICENSE file (MIT/Apache/GPL style), not short notices."""
    a = action.lower()
    if "license notice" in a:
        return False
    # OSADL "Reference License text" = name/URL attribution, not a full copy.
    if "reference" in a:
        return False
    return (
        "license" in a
        and any(v in a for v in ("provide", "forward"))
        and "grant license" not in a
        and "fulfill license" not in a
    )


def _is_source_disclosure_action(action: str) -> bool:
    a = action.lower()
    if "written offer" in a:
        return True
    if "delayed source code" in a:
        return True
    # "Provide File NOTICE In Source code" is not source disclosure.
    patterns = (
        r"\bprovide source code\b",
        r"\breference source code\b",
        r"\binclude source code\b",
        r"\bprovide source code of\b",
        r"\breference source code of\b",
    )
    return any(re.search(p, a) for p in patterns)


def _is_state_changes_action(action: str) -> bool:
    a = action.lower()
    return any(
        k in a
        for k in (
            "modification notice",
            "modification date",
            "legal notices",
            "state changes",
        )
    )


def _is_sublicense_prohibition(action: str) -> bool:
    return "sublicense" in action.lower()


def _is_restrict_granted_rights_ban(action: str) -> bool:
    return "restrict granted rights" in action.lower()


def _is_notice_file_action(action: str) -> bool:
    a = action.lower()
    return 'file "notice"' in a or "file 'notice'" in a


def _is_indemnify_action(action: str) -> bool:
    return action.lower().startswith("indemnify")


def _is_forward_patent_action(action: str) -> bool:
    return "forward patent" in action.lower()


def _is_forward_trademark_action(action: str) -> bool:
    return "forward trademark" in action.lower()


def _is_inform_recipients_action(action: str) -> bool:
    return action.lower().startswith("inform recipients")


def _node_has_delayed_source(node: Any) -> bool:
    if isinstance(node, str):
        return "delayed source code" in node.lower()
    if isinstance(node, dict):
        return any(
            _node_has_delayed_source(k) or _node_has_delayed_source(v) for k, v in node.items()
        )
    if isinstance(node, list):
        return any(_node_has_delayed_source(v) for v in node)
    return False


def _is_file_level_disclosure_action(action: str, rest: Any = None) -> bool:
    if _is_source_disclosure_action(action):
        return True
    if _is_inform_recipients_action(action) and _node_has_delayed_source(rest):
        return True
    return False


def _is_promote_ban(action: str) -> bool:
    a = action.lower()
    return a == "promote" or a.startswith("promote ")


def _is_modify_license_notices_ban(action: str) -> bool:
    a = action.lower()
    return "modify license" in a and "notice" in a


def _is_modify_legal_notices_ban(action: str) -> bool:
    return "modify legal notices" in action.lower()


def _is_anti_circumvention_ban(action: str) -> bool:
    a = action.lower()
    return "circumvention" in a and "technological measures" in a


def _is_modify_license_ban(action: str) -> bool:
    a = action.lower()
    return a == "modify license" or a.startswith("modify license")


def _is_trademark_name_ban(action: str) -> bool:
    a = action.lower()
    return "in product name" in a or "use trademark" in a


def _is_reserved_font_name_ban(action: str) -> bool:
    return "reserved font name" in action.lower()


def _is_sell_font_ban(action: str) -> bool:
    return "sell font" in action.lower()


def _has_font_modification(conditions: tuple[str, ...]) -> bool:
    return any("font modification" in c.lower() for c in conditions)


def _is_gpl3_family(license_id: str) -> bool:
    return license_id.startswith(("GPL-3", "AGPL-3", "LGPL-3"))


def _build_gpl3_hints(license_id: str, obligations: list[ObligationRecord]) -> dict[str, bool]:
    if not _is_gpl3_family(license_id):
        return {}
    hints: dict[str, bool] = {}
    for rec in obligations:
        a = rec.action.lower()
        if rec.polarity == "must" and "installation information" in a:
            hints["installation_info"] = True
        if rec.polarity == "must" and "non-permissive additional terms" in a:
            hints["additional_terms"] = True
        if rec.polarity == "must_not" and _is_anti_circumvention_ban(rec.action):
            hints["anti_circumvention_ban"] = True
    return hints


def _build_linking_obligations(obligations: list[ObligationRecord]) -> dict[str, bool]:
    hints: dict[str, bool] = {}
    for rec in obligations:
        if rec.polarity != "must":
            continue
        a = rec.action.lower()
        if "relink" in a or "linkable work" in a:
            hints["relink"] = True
        if "shared library" in a:
            hints["shared_library"] = True
        if "use gpl-" in a:
            hints["upgrade_to_gpl"] = True
    return hints


def _has_credit_attribution(obligations: list[ObligationRecord]) -> bool:
    return any(
        rec.polarity == "must" and "credit" in rec.action.lower() for rec in obligations
    )


def _has_license_choice(obligations: list[ObligationRecord]) -> bool:
    return any(
        rec.polarity == "must"
        and ("compatible license" in rec.action.lower() or "use original license" in rec.action.lower())
        for rec in obligations
    )


def _is_lgpl_family(license_id: str) -> bool:
    return license_id.startswith("LGPL")


def _has_linking_use_cases(obligations: list[ObligationRecord]) -> bool:
    keys = ("linked work", "combined work", "library")
    return any(
        rec.ctx.use_case and any(k in rec.ctx.use_case.lower() for k in keys)
        for rec in obligations
    )


def _is_license_notice_action(action: str) -> bool:
    """Short licensing / attribution notice (not full license text)."""
    a = action.lower()
    if "license notice" in a and any(v in a for v in ("provide", "forward")):
        return True
    # OSADL shorthand: "Reference License text" / "Reference License text OSL-3.0"
    if "reference" in a and "license" in a and "grant license" not in a and "fulfill license" not in a:
        return True
    return False


def _has_service_offerings(conditions: tuple[str, ...]) -> bool:
    return any("service offerings" in c.lower() for c in conditions)


def _has_software_modification(conditions: tuple[str, ...]) -> bool:
    return any("software modification" in c.lower() for c in conditions)


def _has_network_service_use_case(obligations: list[ObligationRecord]) -> bool:
    return any("saas" in rec.ctx.forms for rec in obligations) or any(
        rec.ctx.use_case and "network service" in rec.ctx.use_case.lower()
        for rec in obligations
    )


def _walk(node: Any, ctx: WalkContext, acc: ProfileAccumulator) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in META_KEYS:
                _ingest_meta(key, value, acc)
                continue
            if key == "USE CASE":
                if isinstance(value, dict):
                    for uc_name, uc_body in value.items():
                        _walk(uc_body, ctx.with_use_case(str(uc_name)), acc)
                elif isinstance(value, str):
                    _walk({}, ctx.with_use_case(value), acc)
                continue
            if key in ("IF", "EXCEPT IF"):
                if isinstance(value, dict):
                    for cond, body in value.items():
                        _walk(body, ctx.with_condition(str(cond)), acc)
                continue
            if key in BRANCH_KEYS:
                _walk(value, ctx, acc)
                continue
            if key == "YOU MUST":
                _ingest_must(value, ctx, acc)
                continue
            if key == "YOU MUST NOT":
                _ingest_must_not(value, ctx, acc)
                continue
            if key == "ATTRIBUTE":
                continue
            _walk(value, ctx, acc)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                continue
            _walk(item, ctx, acc)


def _ingest_meta(key: str, value: Any, acc: ProfileAccumulator) -> None:
    if key == "COPYLEFT CLAUSE":
        val = str(value).strip().lower()
        if val == "yes":
            acc.copyleft = "yes"
        elif val == "questionable":
            acc.copyleft = "questionable"
        else:
            acc.copyleft = acc.copyleft or "no"
    elif key == "PATENT HINTS" and str(value).strip().lower() == "yes":
        acc.patent_hints = True


def _ingest_must(value: Any, ctx: WalkContext, acc: ProfileAccumulator) -> None:
    if isinstance(value, str):
        acc.add_must(value, ctx)
    elif isinstance(value, list):
        for item in value:
            _ingest_must(item, ctx, acc)
    elif isinstance(value, dict):
        for action, rest in value.items():
            acc.add_must(str(action), ctx)
            if _is_inform_recipients_action(str(action)) and _node_has_delayed_source(rest):
                acc.add_must("Provide source code (delayed)", ctx)
            if isinstance(rest, dict):
                _walk(rest, ctx, acc)


def _ingest_must_not(value: Any, ctx: WalkContext, acc: ProfileAccumulator) -> None:
    if isinstance(value, str):
        acc.add_must_not(value, ctx)
    elif isinstance(value, list):
        for item in value:
            _ingest_must_not(item, ctx, acc)
    elif isinstance(value, dict):
        for action, rest in value.items():
            acc.add_must_not(str(action), ctx)
            if isinstance(rest, dict):
                _walk(rest, ctx, acc)


def load_osadl_json_opt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"expected single-license root object: {path}")
    return next(iter(data.values()))


def resolve_osadl_path(license_id: str, *, json_opt_dir: Path = JSON_OPT_DIR) -> Path:
    filename = MG_TO_OSADL_FILE.get(license_id) or f"{license_id}-opt.json"
    path = json_opt_dir / filename
    if not path.is_file():
        alt = json_opt_dir / f"{license_id}-opt.json"
        if alt.is_file():
            return alt
    if not path.is_file():
        raise FileNotFoundError(f"OSADL json-opt not found for {license_id}: {path}")
    return path


def build_profile_from_osadl(
    license_id: str,
    *,
    osadl_tree: dict[str, Any] | None = None,
    osadl_path: Path | None = None,
) -> dict[str, Any]:
    if osadl_tree is None:
        path = osadl_path or resolve_osadl_path(license_id)
        osadl_tree = load_osadl_json_opt(path)
        osadl_file = path.name
    else:
        osadl_file = osadl_path.name if osadl_path else f"{license_id}-opt.json"

    acc = ProfileAccumulator(license_id=license_id, osadl_file=osadl_file)
    _walk(osadl_tree, WalkContext(), acc)
    return _materialize_profile(acc)


def _materialize_profile(acc: ProfileAccumulator) -> dict[str, Any]:
    provides: list[str] = []
    schema_gaps: list[str] = []
    profile: dict[str, Any] = {
        "short_id": acc.license_id,
        "_audit": {
            "osadl_file": acc.osadl_file,
            "profile_version": 4,
            "provides": provides,
            "schema_gaps": schema_gaps,
        },
        "not_provided": [
            "usages",
            "result",
            "relicense",
            "compat",
            "rights",
            "coverage",
            "labels",
        ],
    }

    copyleft_status = acc.copyleft or "no"
    if copyleft_status == "yes":
        profile["copyleft_status"] = "yes"
    elif copyleft_status == "questionable":
        profile["copyleft_status"] = "questionable"
    else:
        profile["copyleft_status"] = "no"
    provides.append("copyleft_status")

    has_notice = any(
        rec.polarity == "must" and _is_notice_action(rec.action) for rec in acc.obligations
    )
    has_license_text = any(
        rec.polarity == "must" and _is_license_text_action(rec.action) for rec in acc.obligations
    )
    has_license_notice = any(
        rec.polarity == "must" and _is_license_notice_action(rec.action) for rec in acc.obligations
    )
    disclose_forms = _disclose_forms(acc.obligations)
    has_source_disclosure = bool(disclose_forms)
    has_sublicense_ban = any(
        rec.polarity == "must_not" and _is_sublicense_prohibition(rec.action)
        for rec in acc.obligations
    )
    has_extra_restrictions_ban = any(
        rec.polarity == "must_not" and _is_restrict_granted_rights_ban(rec.action)
        for rec in acc.obligations
    )
    has_grant_same_license = any(
        _has_software_modification(rec.ctx.conditions) or _has_font_modification(rec.ctx.conditions)
        for rec in acc.grant_license_actions
    )
    has_notice_file = any(
        rec.polarity == "must" and _is_notice_file_action(rec.action) for rec in acc.obligations
    )
    has_indemnify_service = any(
        rec.polarity == "must"
        and _is_indemnify_action(rec.action)
        and _has_service_offerings(rec.ctx.conditions)
        for rec in acc.obligations
    )
    has_forward_patent = any(
        rec.polarity == "must" and _is_forward_patent_action(rec.action) for rec in acc.obligations
    )
    has_forward_trademark = any(
        rec.polarity == "must" and _is_forward_trademark_action(rec.action)
        for rec in acc.obligations
    )
    has_promote_ban = any(
        rec.polarity == "must_not" and _is_promote_ban(rec.action) for rec in acc.obligations
    )
    has_modify_license_notices_ban = any(
        rec.polarity == "must_not" and _is_modify_license_notices_ban(rec.action)
        for rec in acc.obligations
    )
    has_modify_legal_notices_ban = any(
        rec.polarity == "must_not" and _is_modify_legal_notices_ban(rec.action)
        for rec in acc.obligations
    )
    has_anti_circumvention_ban = any(
        rec.polarity == "must_not" and _is_anti_circumvention_ban(rec.action)
        for rec in acc.obligations
    )
    has_modify_license_ban = any(
        rec.polarity == "must_not" and _is_modify_license_ban(rec.action)
        for rec in acc.obligations
    )
    has_trademark_name_ban = any(
        rec.polarity == "must_not" and _is_trademark_name_ban(rec.action)
        for rec in acc.obligations
    )
    has_reserved_font_name_ban = any(
        rec.polarity == "must_not" and _is_reserved_font_name_ban(rec.action)
        for rec in acc.obligations
    )
    has_sell_font_ban = any(
        rec.polarity == "must_not" and _is_sell_font_ban(rec.action)
        for rec in acc.obligations
    )

    categories: dict[str, bool] = {}
    if copyleft_status == "yes":
        categories["copyleft"] = True
    elif copyleft_status == "no":
        categories["copyleft"] = False
    categories["disclose"] = has_source_disclosure
    profile["categories"] = categories
    provides.append("categories")

    if not acc.obligations and copyleft_status == "no":
        profile["zero_obligations"] = True
        provides.append("zero_obligations")

    if disclose_forms:
        profile["disclose_forms"] = disclose_forms
        provides.append("disclose_forms")

    redistribute: dict[str, bool] = {}
    if has_notice or has_license_notice:
        redistribute["include_notice"] = True
    if has_license_text:
        redistribute["include_license"] = True
    if redistribute:
        profile["redistribute"] = redistribute
        provides.append("redistribute")

    redist_slices = _build_redistribute_slices(acc.obligations)
    if redist_slices:
        profile["redistribute_slices"] = redist_slices
        provides.append("redistribute_slices")

    must_not_hints: dict[str, bool] = {}
    if has_promote_ban:
        must_not_hints["promote_ban"] = True
    if has_modify_license_notices_ban:
        must_not_hints["modify_license_notices_ban"] = True
    if has_modify_legal_notices_ban:
        must_not_hints["modify_legal_notices_ban"] = True
    if has_anti_circumvention_ban:
        must_not_hints["anti_circumvention_ban"] = True
    if has_modify_license_ban:
        must_not_hints["modify_license_ban"] = True
    if has_trademark_name_ban:
        must_not_hints["trademark_name_ban"] = True
    if has_reserved_font_name_ban:
        must_not_hints["reserved_font_name_ban"] = True
        schema_gaps.append("font_hints.reserved_font_name")
    if has_sell_font_ban:
        must_not_hints["sell_font_standalone_ban"] = True
        schema_gaps.append("font_hints.standalone_sale")
    if must_not_hints:
        profile["must_not_hints"] = must_not_hints
        provides.append("must_not_hints")

    reserved_hints: dict[str, bool] = {}
    if has_sublicense_ban:
        reserved_hints["sublicense_prohibited"] = True
    if reserved_hints:
        profile["reserved_hints"] = reserved_hints
        provides.append("reserved_hints")

    if has_extra_restrictions_ban:
        profile["prohibit_extra_restrictions"] = True
        provides.append("prohibit_extra_restrictions")
        schema_gaps.append("prohibit_extra_restrictions.no_mg_audit")

    if has_grant_same_license:
        profile["grant_same_license"] = True
        provides.append("grant_same_license")

    if acc.patent_hints:
        profile["patent_hints"] = True
        provides.append("patent_hints")
        if not has_forward_patent:
            schema_gaps.append("patent_hints.metadata_only")

    apache_hints: dict[str, bool] = {}
    if has_notice_file:
        apache_hints["notice_file"] = True
    if has_forward_patent:
        apache_hints["forward_patent"] = True
    if has_forward_trademark:
        apache_hints["forward_trademark"] = True
    if apache_hints:
        profile["apache_hints"] = apache_hints
        provides.append("apache_hints")
        if apache_hints.get("notice_file"):
            profile["redistribute_precision"] = {"no_global_state_changes": True}
            provides.append("redistribute_precision")

    if has_indemnify_service:
        profile["indemnify_on_service"] = True
        provides.append("indemnify_on_service")
        schema_gaps.append("indemnify_on_service")

    if _is_lgpl_family(acc.license_id) and copyleft_status == "yes" and _has_linking_use_cases(
        acc.obligations
    ):
        profile["weak_copyleft"] = True
        profile["linking_hints"] = {"modify_copyleft": True, "combine_copyleft": False}
        provides.extend(["weak_copyleft", "linking_hints"])
        linking_obligations = _build_linking_obligations(acc.obligations)
        if linking_obligations:
            profile["linking_obligations"] = linking_obligations
            provides.append("linking_obligations")

    gpl3_hints = _build_gpl3_hints(acc.license_id, acc.obligations)
    if gpl3_hints:
        profile["gpl3_hints"] = gpl3_hints
        provides.append("gpl3_hints")
        if gpl3_hints.get("installation_info"):
            schema_gaps.append("gpl3_hints.installation_info")
        if gpl3_hints.get("additional_terms"):
            schema_gaps.append("gpl3_hints.additional_terms")
        if gpl3_hints.get("anti_circumvention_ban"):
            schema_gaps.append("gpl3_hints.anti_circumvention")

    if _has_credit_attribution(acc.obligations):
        profile["attribution_hints"] = {"require_credit": True}
        provides.append("attribution_hints")

    if _has_license_choice(acc.obligations):
        profile["license_choice"] = True
        provides.append("license_choice")

    if acc.license_id == "OFL-1.1":
        profile["font_delivery"] = True
        provides.append("font_delivery")

    slices = _build_terms_slices(acc.obligations)
    if slices:
        profile["terms_slices"] = slices
        provides.append("terms_slices")

    if _has_network_service_use_case(acc.obligations):
        profile["network_service"] = True
        provides.append("network_service")

    profile["_audit"]["provides"] = provides
    profile["_audit"]["schema_gaps"] = schema_gaps
    return profile


def _disclose_forms(obligations: list[ObligationRecord]) -> list[str]:
    forms: set[str] = set()
    for rec in obligations:
        if rec.polarity != "must":
            continue
        if _is_source_disclosure_action(rec.action) or rec.action.lower().startswith(
            "provide source code (delayed)"
        ):
            forms.update(rec.ctx.forms)
    order = ["raw", "binary", "saas"]
    return [f for f in order if f in forms]


def _build_terms_slices(obligations: list[ObligationRecord]) -> list[dict[str, Any]]:
    # (forms, when, kind, tag) -> present
    # kind: restrictions | obligations
    merged: dict[tuple[tuple[str, ...], tuple[str, ...], str, str], dict[str, Any]] = {}

    for rec in obligations:
        if rec.polarity != "must" or not rec.ctx.forms:
            continue
        forms_key = tuple(sorted(rec.ctx.forms))
        when_key = tuple(sorted(c for c in (_cond_label(x) for x in rec.ctx.conditions) if c))

        tags: list[tuple[str, str]] = []
        if _is_state_changes_action(rec.action) and _has_software_modification(rec.ctx.conditions):
            tags.append(("restrictions", "state_changes"))
        if _is_source_disclosure_action(rec.action) and _has_software_modification(rec.ctx.conditions):
            tags.append(("obligations", "source_disclosure"))
        if (
            rec.action.lower().startswith("provide source code (delayed)")
            and "binary" in rec.ctx.forms
        ):
            tags.append(("obligations", "source_disclosure"))

        for kind, tag in tags:
            key = (forms_key, when_key, kind, tag)
            if key in merged:
                continue
            item: dict[str, Any] = {"forms": list(forms_key)}
            if when_key:
                item["when"] = list(when_key)
            if kind == "restrictions":
                item["restrictions"] = [tag]
            else:
                item["obligations"] = [tag]
            merged[key] = item

    # merge rows with same forms+when into one slice
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    for (forms_key, when_key, kind, tag), item in sorted(merged.items()):
        gkey = (forms_key, when_key)
        row = grouped.setdefault(
            gkey,
            {"forms": list(forms_key), **({"when": list(when_key)} if when_key else {})},
        )
        if kind == "restrictions":
            row.setdefault("restrictions", [])
            if tag not in row["restrictions"]:
                row["restrictions"].append(tag)
        else:
            row.setdefault("obligations", [])
            if tag not in row["obligations"]:
                row["obligations"].append(tag)

    out: list[dict[str, Any]] = []
    for forms_key, when_key in sorted(grouped):
        row = grouped[(forms_key, when_key)]
        if not row.get("restrictions") and not row.get("obligations"):
            continue
        if "restrictions" in row:
            row["restrictions"] = sorted(row["restrictions"])
        if "obligations" in row:
            row["obligations"] = sorted(row["obligations"])
        out.append(row)
    return out


def _cond_label(cond: str) -> str | None:
    c = cond.lower()
    if "software modification" in c:
        return "software_modification"
    if "network service" in c:
        return "network_service"
    if "font modification" in c:
        return "font_modification"
    return None


def _build_redistribute_slices(obligations: list[ObligationRecord]) -> list[dict[str, Any]]:
    per_form: dict[str, dict[str, bool]] = {}
    for rec in obligations:
        if rec.polarity != "must" or not rec.ctx.forms:
            continue
        for form in rec.ctx.forms:
            row = per_form.setdefault(form, {})
            if _is_notice_action(rec.action) or _is_license_notice_action(rec.action):
                row["include_notice"] = True
            if _is_license_text_action(rec.action):
                row["include_license"] = True
    out: list[dict[str, Any]] = []
    for form in ("raw", "binary", "saas"):
        row = per_form.get(form)
        if not row:
            continue
        item: dict[str, Any] = {"forms": [form]}
        for key in ("include_notice", "include_license"):
            if row.get(key):
                item[key] = True
        if len(item) > 1:
            out.append(item)
    return out


def write_profile(profile: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def list_profile_ids(*, profile_dir: Path = DEFAULT_PROFILE_DIR) -> list[str]:
    return sorted(p.stem for p in profile_dir.glob("*.yaml") if p.is_file())


def profile_obligation_signature(profile: dict[str, Any]) -> tuple[Any, ...]:
    """Coarse obligation-family key for coverage dedup checks."""
    cats = profile.get("categories", {})
    red = profile.get("redistribute", {})
    return (
        profile.get("copyleft_status"),
        cats.get("copyleft"),
        cats.get("disclose"),
        tuple(sorted(red.keys())),
        profile.get("grant_same_license"),
        tuple(sorted(profile.get("disclose_forms") or [])),
        profile.get("network_service"),
        bool(profile.get("apache_hints")),
        len(profile.get("terms_slices") or []),
        bool(profile.get("reserved_hints")),
        bool(profile.get("weak_copyleft")),
    )


def build_profiles(
    license_ids: list[str],
    *,
    json_opt_dir: Path = JSON_OPT_DIR,
    out_dir: Path = DEFAULT_PROFILE_DIR,
) -> list[Path]:
    written: list[Path] = []
    for lic_id in license_ids:
        path = resolve_osadl_path(lic_id, json_opt_dir=json_opt_dir)
        profile = build_profile_from_osadl(lic_id, osadl_path=path)
        out_path = out_dir / f"{lic_id}.yaml"
        write_profile(profile, out_path)
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OSADL audit profiles from json-opt checklists")
    parser.add_argument(
        "--license",
        action="append",
        dest="licenses",
        help="License short_id (repeatable). Default: all mapped in MG_TO_OSADL_FILE",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Build curated OSADL_COVERAGE_PROFILES set (~20 licenses)",
    )
    parser.add_argument("--json-opt-dir", type=Path, default=JSON_OPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    args = parser.parse_args(argv)

    if args.coverage:
        license_ids = list(OSADL_COVERAGE_PROFILES)
    else:
        license_ids = args.licenses or sorted(set(MG_TO_OSADL_FILE.keys()))
    paths = build_profiles(license_ids, json_opt_dir=args.json_opt_dir, out_dir=args.out_dir)
    for path in paths:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
