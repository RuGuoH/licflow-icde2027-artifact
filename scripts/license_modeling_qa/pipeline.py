"""QA 建模编排。

Phase 1：词表抽取 → 问卷（L0 + Combine + 可选 modify）→ 装配 YAML。
失败回退 get_license_data_legacy。不得改动 legacy 实现。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .assemble import assemble
from .ask import fill_ir, resolve_named_licenses, run_question_calls
from .embed_recall import (
    embed_supplement_enabled,
    supplement_cues_with_embed,
    supplement_evidence_with_embed,
)
from .extract import (
    extract_cues,
    extract_evidence,
    extract_identity,
    is_short_fulltext,
    openrail_family_incompat,
)
from .ir import LegalIR
from .data_confirm import confirm_data_cue
from .downstream_confirm import confirm_downstream_cue
from .nd_confirm import confirm_nd_cue
from .packs import attach_packs
from .schema import validate_license_data


def run_qa_modeling(helper: Any, license_name: str) -> Optional[dict]:
    """qa 后端入口：抽取 + 问卷 + 装配；失败则回退 legacy。"""
    try:
        return _run_qa(helper, license_name)
    except Exception:
        logging.exception("qa 建模异常，回退 legacy license=%s", license_name)
        return helper.get_license_data_legacy(license_name)


def _run_qa(helper: Any, license_name: str) -> Optional[dict]:
    license_text = helper.fetch_license_text(license_name)
    if not license_text or license_text.strip() == str(license_name).strip():
        logging.info("qa 无全文，回退 legacy license=%s", license_name)
        return helper.get_license_data_legacy(license_name)

    logging.info("LICENSE_MODELING_PIPELINE=qa（phase 1） license=%s", license_name)
    evidence = extract_evidence(license_text)
    if embed_supplement_enabled() and not is_short_fulltext(license_text):
        evidence = supplement_evidence_with_embed(license_text, evidence)
    cues = extract_cues(license_text)
    if embed_supplement_enabled() and not is_short_fulltext(license_text):
        cues = supplement_cues_with_embed(license_text, cues)
    # 点名许可和三条确认互不读答案，一起问。compat 问卷是否发出要等点名结果。
    preface = run_question_calls(
        [
            ("named", lambda: resolve_named_licenses(helper, license_name, license_text)),
            ("nd", lambda: confirm_nd_cue(helper, license_name, license_text, cues)),
            ("data", lambda: confirm_data_cue(helper, license_name, license_text, cues)),
            (
                "downstream",
                lambda: confirm_downstream_cue(helper, license_name, license_text, cues),
            ),
        ]
    )
    named = preface["named"]
    packs = attach_packs(
        evidence, license_text, named_from_text=bool(named.allow or named.ban)
    )
    ir = LegalIR(
        identity=extract_identity(license_text, license_name),
        cues=cues,
        named=named,
    )
    ir = fill_ir(helper, license_name, ir, evidence, packs, license_text)

    if not ir.l0_ok and not ir.combine_ok:
        logging.warning("qa_fallback_legacy reason=no_json license=%s", license_name)
        return helper.get_license_data_legacy(license_name)

    data = assemble(license_name, ir)
    family = openrail_family_incompat(license_text)
    if family and not data.get("incompat"):
        data["incompat"] = family
        if not data.get("compat"):
            data.pop("compat", None)
    errors = validate_license_data(data)
    if errors:
        logging.warning(
            "qa_fallback_legacy reason=schema license=%s errors=%s",
            license_name,
            errors,
        )
        return helper.get_license_data_legacy(license_name)

    if hasattr(helper, "save_revision_for_inspection"):
        helper.save_revision_for_inspection(
            license_name, data, "license_final", 0, extra_label="qa"
        )
    return data
