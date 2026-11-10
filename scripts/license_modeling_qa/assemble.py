"""法律事实 IR → license_data。唯一知道 ModelGo 字段名的模块。"""
from __future__ import annotations

import re
from typing import Any, Optional

from .extract import normalize_named_license
from .ir import (
    AugmentFacts,
    CombineFacts,
    CombineSituation,
    EmbedFacts,
    GrantFacts,
    LegalIR,
    ModifyFacts,
    OutputFacts,
    RestrictFacts,
    StatFacts,
    TextCues,
    TrainFacts,
    resolved_work_types,
)
from .types import EvidenceItem

_NOTICE_MAP = {
    "notice": "include_notice",
    "license_text": "include_license",
    "state_changes": "state_changes",
}
_DEFAULT_FORMS = ["raw", "binary"]
_AI_USAGES = ["train", "distill", "generate", "embed", "augment", "stat"]


def assemble(license_name: str, ir: LegalIR) -> dict[str, Any]:
    ident = ir.identity
    grant = ir.grant
    categories = _categories(ir)
    layer_a, layer_b, spread_c, use_c = _restriction_layers(ir)
    redistribute = _dedupe(layer_a + [x for x in spread_c if x == "include_use_restriction"])
    compat, incompat = _named_license_lists(license_name, ir)
    terms = _apply_your_derivative_relicense(
        _apply_other_kind_exception(
            _apply_embed_inherits_modify(
                _apply_amalgamate_state_changes(
                    _apply_distill_named_restriction(_terms(ir, compat), ir), ir
                ),
                ir,
            ),
            ir,
        ),
        ir,
    )
    rights, reserved = _rights(
        grant, ir.combine, ir.cues, terms=terms, evidence=ir.evidence
    )
    coverage = _coverage(grant, ir.modify, terms, ir.cues)
    if _is_obligation_thin(ir.cues):
        # 公有领域 / 极弱义务：无再分发通知骨架、无 coverage 切片
        redistribute = []
        coverage = []
    if ir.cues.purpose_use_passes_on and not _is_obligation_thin(ir.cues):
        if "include_use_restriction" not in redistribute:
            redistribute.append("include_use_restriction")
        for row in terms:
            if "use" not in (row.get("usages") or []):
                continue
            restrictions = list(row.get("restrictions") or [])
            if "include_use_restriction" not in restrictions:
                restrictions.append("include_use_restriction")
            row["restrictions"] = restrictions
    url = ident.url or f"https://example.com/license/{license_name}"
    return {
        "full_name": ident.full_name or license_name,
        "short_id": license_name,
        "url": url,
        "version": ident.version,
        "available": True,
        "categories": categories,
        "labels": [],
        "rights": rights,
        "reserved_rights": reserved,
        "rights_prefix": list(ir.cues.rights_prefix or []),
        "coverage": coverage,
        "redistribute": redistribute,
        "compat": compat,
        "incompat": incompat,
        "terms": terms,
    }


def _aup_pass_on(ir: LegalIR) -> bool:
    restrict = ir.restrict
    return bool(restrict and restrict.mentioned and restrict.pass_on != "no")


_DISCLOSE_EVIDENCE_RE = re.compile(
    r"Corresponding Source|"
    r"Executable Form.{0,200}Source Code Form|"
    r"must also be made available as Source Code|"
    r"provide recipients? the source code|"
    r"how they can obtain a copy of such Source Code|"
    r"(?:object code|binary(?:\s+form)?|Executable Form).{0,160}"
    r"(?:must|shall)\b.{0,100}"
    r"(?:(?:also )?be made available|provide|accompany|offer|inform).{0,100}"
    r"(?:source code|Source Code(?: Form)?|Corresponding Source)",
    re.IGNORECASE | re.DOTALL,
)
_DISCLOSE_FALSE_FRIEND_RE = re.compile(
    r"Redistribution and use in source and binary forms|"
    r"Redistributions of source code must retain|"
    r"Redistributions in binary form must reproduce|"
    r"in source and binary forms, with or without",
    re.IGNORECASE,
)


def _requires_disclose(ir: LegalIR) -> bool:
    """证据表明目标码/可执行形态分发须提供源码时抬 disclose。"""
    if ir.grant.disclose == "yes":
        return True
    blob = " ".join((e.quote or "") for e in (ir.evidence or []))
    if not blob:
        return False
    if not _DISCLOSE_EVIDENCE_RE.search(blob):
        return False
    strong = re.search(
        r"Corresponding Source|provide recipients? the source code|"
        r"must also be made available as Source Code|"
        r"how they can obtain a copy of such Source Code|"
        r"(?:must|shall)\b.{0,100}(?:source code|Source Code|Corresponding Source)",
        blob,
        re.IGNORECASE | re.DOTALL,
    )
    if _DISCLOSE_FALSE_FRIEND_RE.search(blob) and not strong:
        return False
    return bool(
        re.search(
            r"distribut|convey|Executable Form|object code|binary|made available|"
            r"provide recipients|inform recipients",
            blob,
            re.IGNORECASE,
        )
    )


def _is_ultra_thin(cues: TextCues) -> bool:
    """Unlicense / WTFPL 式：几乎无版权义务骨架。"""
    if cues.ultra_permissive:
        return True
    if not cues.waive_copyright:
        return False
    # 有专利/商标明示保留或结构化通知义务 → 走放弃后的 independent，而非空 NODEF。
    if cues.patent_denied or cues.trademark_use_denied:
        return False
    if cues.attach_license or cues.retain_notices:
        return False
    return True


def _is_waiver_independent(cues: TextCues) -> bool:
    """CC0 / PDDL 式：版权放弃但仍可建模为无义务 independent。"""
    return bool(cues.waive_copyright) and not _is_ultra_thin(cues)


def _is_obligation_thin(cues: TextCues) -> bool:
    return _is_ultra_thin(cues) or _is_waiver_independent(cues)


def _adapted_relicense(
    copyleft: bool,
    stated: str,
    compat: Optional[list[str]] = None,
    aup_pass: bool = False,
    different_terms: bool = False,
) -> Any:
    if copyleft:
        return _norm_relicense(
            stated, default=_copyleft_relicense(compat or [])
        )
    if aup_pass:
        return "conditional"
    if different_terms:
        return True
    return _norm_relicense(stated, default=True)


def _allows_modification_or_translation(ir: LegalIR) -> bool:
    """modification/translation：数据再利用证据或 data/mixed；不硬绑 categories:data。

    显式 software 且无 DATA_REUSE topic 时，即使误填 embed/augment 也降级。
    signals 未定案时，允许已填的 embed/augment（单测与短 IR）。
    """
    from .gazetteer import DATA_REUSE_TOPICS

    if any(e.topic in DATA_REUSE_TOPICS for e in ir.evidence):
        return True
    types = resolved_work_types(ir.grant.work_signals, ir.cues)
    if "data" in types:
        return True
    mentioned = bool(
        (ir.augment and ir.augment.mentioned) or (ir.embed and ir.embed.mentioned)
    )
    if types == ["software"]:
        return False
    return mentioned


def _sublicense_in_rights(ir: LegalIR) -> bool:
    """明示授予才算权利里有 sublicense。禁止和沉默都不是。"""
    cues = ir.cues
    if cues.sublicense_forbidden:
        return False
    if cues.sublicense_granted or cues.grant_sublicense:
        return True
    return "sublicense" in (ir.grant.grants or [])


def _categories(ir: LegalIR) -> list[str]:
    grant = ir.grant
    cats: list[str] = []
    if grant.public == "yes":
        cats.append("public")
    types = resolved_work_types(grant.work_signals, ir.cues)
    for kind in ("software", "data", "model"):
        if kind in types and kind not in cats:
            cats.append(kind)
    copyleft = _scope_copyleft(ir)
    if copyleft:
        cats.append("copyleft")
    # 自动再许可要有自己的句子，或一条没被关掉的兼容许可。同许可本身不加。
    if ir.cues.auto_downstream_grant or (copyleft and _open_compat(ir)):
        cats.append("auto-relicensing")
    if grant.disclose == "yes" or _requires_disclose(ir):
        cats.append("disclose")
    # 放弃版权优先，不标 proprietary。
    # 收回商业使用、收回修改件分发，或商业使用本身受限制且未授予 sublicense。
    if not _is_obligation_thin(ir.cues) and (
        ir.cues.non_commercial
        or ir.cues.nd_no_redistribute
        or (ir.cues.commercial_use_limited and not _sublicense_in_rights(ir))
    ):
        cats.append("proprietary")
    if (
        "proprietary" not in cats
        and not copyleft
        and grant.public == "yes"
        and "redistribute" in (grant.grants or [])
        and not _aup_pass_on(ir)
    ):
        cats.append("permissive")
    # 极弱义务：有再分发权能或放弃版权时仍标 permissive
    if (
        "permissive" not in cats
        and "proprietary" not in cats
        and not copyleft
        and _is_obligation_thin(ir.cues)
    ):
        cats.append("permissive")
    return _dedupe(cats)


def _scope_copyleft(ir: LegalIR) -> bool:
    """同许可档位锁定类别。未锁定时仍看问卷的 same_license。"""
    scope = ir.cues.same_license_scope or "none"
    if scope in {"whole", "file_all_forms"}:
        return True
    if scope == "source_only":
        # 附上许可文本就满足的源码义务，不是对整个衍生作品的传染。
        return not ir.cues.source_same_by_attaching
    grant = ir.grant
    modify = ir.modify
    return grant.same_license_this == "yes" or (
        modify is not None and modify.same_license == "yes"
    )


def _open_compat(ir: LegalIR) -> bool:
    banned = {item.lower() for item in (ir.named.ban or [])}
    names = list(ir.named.allow or []) + list(ir.grant.compat_names or [])
    return any(item and item.lower() not in banned for item in names)


def _named_license_lists(self_id: str, ir: LegalIR) -> tuple[list[str], list[str]]:
    allow: list[str] = list(ir.named.allow or [])
    ban: list[str] = list(ir.named.ban or [])
    for raw in ir.grant.compat_names or []:
        sid = normalize_named_license(str(raw))
        if sid:
            allow.append(sid)
    me = (self_id or "").strip().lower()
    rejected = {item.lower() for item in (ir.named.rejected or [])}
    allow = _dedupe(
        [x for x in allow if x and x.lower() != me and x.lower() not in rejected]
    )
    ban = _dedupe([x for x in ban if x and x.lower() != me and x not in allow])
    return allow, ban


def _rights(
    grant: GrantFacts,
    combine: CombineFacts,
    cues: Optional[TextCues] = None,
    terms: Optional[list[dict[str, Any]]] = None,
    evidence: Optional[list[EvidenceItem]] = None,
) -> tuple[list[str], list[str]]:
    allowed = {
        "use",
        "modify",
        "merge",
        "redistribute",
        "sublicense",
        "commercial_use",
        "patent_use",
        # copyright 是所有权原子：仅 waive/dedication 才进 rights，不经 grants
    }
    reserved_allowed = {
        "trademark_use",
        "copyright",
        "patent",
        "trademark",
        "sublicense",
        "commercial_use",
        "patent_use",
        "redistribute",
    }
    cues = cues or TextCues()
    rights = [g for g in grant.grants if g in allowed and g != "patent_use"]
    # 版权授予句权能回填（reproduce→use, prepare→modify, …）
    if cues.grant_use and "use" not in rights:
        rights.append("use")
    if cues.grant_modify and "modify" not in rights:
        rights.append("modify")
    if cues.grant_redistribute and "redistribute" not in rights:
        rights.append("redistribute")
    if cues.grant_sublicense and "sublicense" not in rights:
        rights.append("sublicense")
    if cues.grant_merge and "merge" not in rights:
        rights.append("merge")
    # 已建模的活动行弱回写（terms 有 modify/use 则 rights 对齐）
    for tok in _rights_from_terms(terms):
        if tok in allowed and tok not in rights and tok != "patent_use":
            rights.append(tok)
    # merge = 合入他作的权能，与 combine.result（independent/derivative）解耦：
    # 独立 combine、已建模 combine 行、或制备+再分发授予均可回填。
    if "merge" not in rights and _merge_capability(combine, cues, terms, rights):
        rights.append("merge")
    reserved = [r for r in grant.reserved if r in reserved_allowed and r != "patent_use"]

    if cues.non_commercial:
        rights = [r for r in rights if r != "commercial_use"]
        if "commercial_use" not in reserved:
            reserved.append("commercial_use")
    elif "commercial_use" not in reserved and (
        "use" in rights or "redistribute" in rights
    ):
        rights.append("commercial_use")

    types = resolved_work_types(grant.work_signals, cues)
    data_like = "data" in types and "software" not in types and "model" not in types
    if data_like:
        rights = [r for r in rights if r != "patent_use"]
        if "patent_use" not in reserved:
            reserved.append("patent_use")
    elif cues.patent_grant or "patent_use" in (grant.grants or []):
        # 正向授予（cue 或 L0）优先于误伤性 deny；与 use/modify 解耦
        if "patent_use" not in rights:
            rights.append("patent_use")
        reserved = [r for r in reserved if r != "patent_use"]
    elif cues.patent_denied:
        rights = [r for r in rights if r != "patent_use"]
        if "patent_use" not in reserved:
            reserved.append("patent_use")

    if cues.sublicense_forbidden:
        rights = [r for r in rights if r != "sublicense"]
        if "sublicense" not in reserved:
            reserved.append("sublicense")
    elif cues.sublicense_granted and "sublicense" not in rights:
        rights.append("sublicense")
        reserved = [r for r in reserved if r != "sublicense"]
    elif (
        cues.sublicense_distribution_only
        and not cues.grant_sublicense
        and not cues.sublicense_granted
    ):
        # 换证句里的 sublicense 不是把权利再授出去。
        rights = [r for r in rights if r != "sublicense"]
    elif (
        "non-sublicensable" in (cues.rights_prefix or [])
        and not cues.waive_copyright
        and "sublicense" not in rights
        and "sublicense" not in reserved
    ):
        # 与金标准 CC 等习惯对齐：non-sublicensable 前缀 ⇒ reserved.sublicense
        # 公有领域放弃（CC0）不双写。
        reserved.append("sublicense")

    # copyright：默认保留；仅作品级放弃/公有领域才进 rights
    rights = [r for r in rights if r != "copyright"]
    if cues.waive_copyright:
        reserved = [r for r in reserved if r != "copyright"]
        rights.append("copyright")
    elif "copyright" not in reserved:
        reserved.append("copyright")
    if _is_ultra_thin(cues):
        # 极弱许可：沉默不补 patent/trademark；有明示 deny 再写。
        reserved = [
            r
            for r in reserved
            if r
            not in {
                "patent",
                "trademark",
                "patent_use",
                "trademark_use",
                "copyright",
            }
        ]
        if cues.patent_denied:
            reserved.append("patent")
            if "patent_use" not in rights:
                reserved.append("patent_use")
        for tok in ("use", "modify", "merge", "redistribute"):
            if tok not in rights:
                rights.append(tok)
        if "commercial_use" not in rights and "commercial_use" not in reserved:
            rights.append("commercial_use")
    else:
        # 专利权和商标权沉默默认保留。所有权被彻底放弃的极弱许可不走这里。
        # 正文把商标权本身授出时，不再保留。
        for item in ("patent", "trademark"):
            if item not in rights and item not in reserved:
                reserved.append(item)
        if "trademark" in (grant.grants or []):
            reserved = [r for r in reserved if r != "trademark"]
    # ND 只收紧 coverage，不把 redistribute 挪进保留权利。
    # 未修改原件仍可再分发；改编版靠 coverage 只留 duplicate 来禁止共享。
    # trademark_use：cue 明示否认；或 L0 勾选且证据可核。沉默 L0 不保留。
    if _keep_trademark_use(grant, cues, evidence):
        if "trademark_use" not in rights and "trademark_use" not in reserved:
            reserved.append("trademark_use")
    else:
        reserved = [r for r in reserved if r != "trademark_use"]

    right_set = set(rights)
    reserved = [r for r in reserved if r not in right_set]
    return _dedupe(rights), _dedupe(reserved)


_TRADEMARK_USE_EVIDENCE_RE = re.compile(
    r"no trademark licen[cs]e|"
    r"trademark rights are not licen[cs]ed|"
    r"does not grant.{0,80}trademark|"
    r"(?:must )?not be used to (?:endorse or )?promote|"
    r"shall not be used to (?:promote|endorse|advertise)|"
    r"neither the name.{0,120}endorse or promote|"
    r"\bReserved Font Name\b|"
    r"\bReserved Name\b|"
    r"nothing in this (?:licen[cs]e|agreement) grants you any rights to use"
    r".{0,80}trademark|"
    r"has not been granted any trademark licen[cs]e|"
    r"(?:publicity|advertising|marketing).{0,40}trademark|"
    r"trademark.{0,40}(?:publicity|advertising|marketing)",
    re.IGNORECASE | re.DOTALL,
)


def _keep_trademark_use(
    grant: GrantFacts,
    cues: TextCues,
    evidence: Optional[list[EvidenceItem]],
) -> bool:
    if cues.trademark_use_denied:
        return True
    if "trademark_use" not in (grant.reserved or []):
        return False
    blob = " ".join((e.quote or "") for e in (evidence or []))
    return bool(blob and _TRADEMARK_USE_EVIDENCE_RE.search(blob))


def _rights_from_terms(terms: Optional[list[dict[str, Any]]]) -> list[str]:
    if not terms:
        return []
    found: list[str] = []
    for row in terms:
        if not isinstance(row, dict):
            continue
        if str(row.get("result") or "").upper() == "NODEF":
            continue
        us = row.get("usages") or []
        if "use" in us or "copy" in us:
            found.append("use")
        if "modify" in us:
            found.append("modify")
        # 已建模的 combine（含 AUP→derivative）即合入权能，不要求 independent
        if "combine" in us and row.get("result") in {"independent", "derivative"}:
            found.append("merge")
    return found


def _has_independent_combine_term(terms: Optional[list[dict[str, Any]]]) -> bool:
    if not terms:
        return False
    for row in terms:
        if not isinstance(row, dict):
            continue
        if "combine" not in (row.get("usages") or []):
            continue
        if row.get("result") == "independent":
            return True
    return False


def _has_defined_combine_term(terms: Optional[list[dict[str, Any]]]) -> bool:
    if not terms:
        return False
    for row in terms:
        if not isinstance(row, dict):
            continue
        if "combine" not in (row.get("usages") or []):
            continue
        if row.get("result") in {"independent", "derivative"}:
            return True
    return False


def _merge_capability(
    combine: CombineFacts,
    cues: TextCues,
    terms: Optional[list[dict[str, Any]]],
    rights: list[str],
) -> bool:
    """合入他作权能：与 combine 是否 independent 无关。"""
    if _combine_independent(combine) or _has_independent_combine_term(terms):
        return True
    if _has_defined_combine_term(terms):
        return True
    has_modify = "modify" in rights or cues.grant_modify
    has_redistribute = "redistribute" in rights or cues.grant_redistribute
    return bool(has_modify and has_redistribute)


def _combine_independent(combine: CombineFacts) -> bool:
    if not combine.mentioned:
        return False
    if combine.other_work_must_same_license == "no":
        return True
    return any(s.id == "agg" for s in combine.situations)


def _redistribute(grant: GrantFacts, cues: Optional[TextCues] = None) -> list[str]:
    """A 层：再分发通知。state_changes 不进顶层，落到 derivative 行。"""
    out: list[str] = []
    for item in grant.notice:
        if item == "state_changes":
            continue
        mapped = _NOTICE_MAP.get(item)
        if mapped:
            out.append(mapped)
    cues = cues or TextCues()
    if cues.attach_license:
        out.append("include_license")
    if cues.retain_notices:
        out.append("include_notice")
    return _dedupe(out)


def _restriction_layers(
    ir: LegalIR,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """A=copy 通知；B=state_changes；C=AUP。spread_c 跟再分发走，use_c 只约束使用。"""
    layer_a = _redistribute(ir.grant, ir.cues)
    layer_b: list[str] = []
    if "state_changes" in (ir.grant.notice or []) or ir.cues.mark_changes:
        layer_b.append("state_changes")
    if ir.modify is not None:
        for item in _norm_restrictions(ir.modify.restrictions):
            if item == "state_changes":
                layer_b.append(item)
    use_c, spread_c = _layer_c(ir.restrict)
    return layer_a, _dedupe(layer_b), spread_c, use_c


def _layer_c(restrict: Optional[RestrictFacts]) -> tuple[list[str], list[str]]:
    if restrict is None or not restrict.mentioned:
        return [], []
    use_c: list[str] = ["include_use_restriction"]
    spread_c: list[str] = []
    if restrict.pass_on != "no":
        spread_c.append("include_use_restriction")
    if restrict.runtime == "yes":
        use_c.append("include_runtime_restriction")
        if restrict.pass_on != "no":
            spread_c.append("include_runtime_restriction")
    return _dedupe(use_c), _dedupe(spread_c)


def _saas_enabled(ir: LegalIR) -> bool:
    net = ir.network
    if net is None or not net.mentioned:
        return False
    if net.remote_same_as_convey in {"yes", "only_if_modified"}:
        return True
    blob = " ".join(
        (e.quote or "").lower()
        for e in ir.evidence
        if e.topic == "network_saas"
    )
    return any(
        cue in blob
        for cue in (
            "hosted service",
            "api-based",
            "remotely accessible",
            "as a service",
            "over a network",
            "as a distribution",
            "external deployment",
            "application intended for use over a network",
        )
    )


def _with_saas(forms: list[str], saas: bool) -> list[str]:
    if not saas:
        return _dedupe(forms)
    if "saas" in forms:
        return _dedupe(forms)
    return _dedupe(list(forms) + ["saas"])


def _coverage(
    grant: GrantFacts,
    modify: Optional[ModifyFacts],
    terms: Optional[list[dict[str, Any]]] = None,
    cues: Optional[TextCues] = None,
) -> list[str]:
    if cues and cues.nd_no_redistribute:
        return ["duplicate"]
    cov: list[str] = []
    if "use" in grant.grants or "redistribute" in grant.grants or not grant.grants:
        cov.append("duplicate")
    if grant.same_license_this == "yes" or (
        modify and modify.bound == "bound"
    ):
        cov.append("derivative")
    results = {t.get("result") for t in (terms or [])}
    if "modification" in results:
        cov.append("modification")
    if "translation" in results:
        cov.append("translation")
    return _dedupe(cov)


def _form_split_copy_restrictions(
    ir: LegalIR, shared: list[str]
) -> tuple[list[str], list[str]]:
    """原文区分 source vs binary 附证义务时：binary 可去掉 include_license。

    典型：源码须附完整许可文本，目标码仅须合规许可/声明，全局仍保留 notice。
    无区分句 → raw/binary 同套（保守，防误伤 MIT/Apache）。
    """
    if "include_license" not in shared:
        return shared, shared
    if not ir.cues.binary_skips_attach_license:
        return shared, shared
    raw = list(shared)
    binary = [r for r in shared if r != "include_license"]
    return raw, binary


def _terms(ir: LegalIR, compat: Optional[list[str]] = None) -> list[dict[str, Any]]:
    saas = _saas_enabled(ir)
    use_forms = _with_saas(list(_DEFAULT_FORMS), saas)
    if _is_ultra_thin(ir.cues):
        return [
            {
                "usages": ["use", "copy", "modify"],
                "forms": use_forms,
                "result": "NODEF",
            }
        ]
    if _is_waiver_independent(ir.cues):
        return _terms_waiver_independent(ir, use_forms)

    terms: list[dict[str, Any]]
    layer_a, layer_b, spread_c, use_c = _restriction_layers(ir)
    copy_restr = _dedupe(layer_a + spread_c)
    raw_restr, bin_restr = _form_split_copy_restrictions(ir, copy_restr)
    if layer_a or use_c:
        terms = [
            _term(
                ["use"],
                use_forms,
                "duplicate",
                relicense=False,
                copyleft=False,
                restrictions=use_c,
            ),
        ]
        if raw_restr != bin_restr:
            # 原文区分 source vs object/binary 通知义务时拆行（如 MS-PL）
            terms.append(
                _term(
                    ["copy"],
                    _with_saas(["raw"], saas),
                    "duplicate",
                    relicense=False,
                    copyleft=False,
                    restrictions=raw_restr,
                )
            )
            terms.append(
                _term(
                    ["copy"],
                    _with_saas(["binary"], saas),
                    "duplicate",
                    relicense=False,
                    copyleft=False,
                    restrictions=bin_restr,
                )
            )
        else:
            terms.append(
                _term(
                    ["copy"],
                    use_forms,
                    "duplicate",
                    relicense=False,
                    copyleft=False,
                    restrictions=copy_restr,
                )
            )
    else:
        terms = [
            _term(
                ["use", "copy"],
                use_forms,
                "duplicate",
                relicense=False,
                copyleft=False,
                restrictions=[],
            )
        ]
    aup_pass = _aup_pass_on(ir)
    work_copyleft = _scope_copyleft(ir)
    content_shape = _allows_modification_or_translation(ir)
    if ir.modify is not None:
        terms.extend(
            _modify_terms_for_scope(
                ir,
                layer_b,
                spread_c,
                saas,
                compat or [],
                content_shape,
                aup_pass,
            )
        )
    terms.extend(
        _combine_terms(
            ir.combine,
            ir.evidence,
            compat or [],
            layer_b,
            spread_c,
            saas,
            work_copyleft=work_copyleft,
            aup_pass=aup_pass,
            default_independent=(
                not _is_obligation_thin(ir.cues)
                and "redistribute" in (ir.grant.grants or [])
            ),
        )
    )
    terms.extend(
        _ai_terms(
            ir,
            use_forms,
            layer_b,
            spread_c,
            compat or [],
            aup_pass=aup_pass,
            allow_content_results=content_shape,
        )
    )
    _clone_amalgamate_from_modify(terms, ir, saas, work_copyleft)
    _align_amalgamate_with_modify(terms, ir, work_copyleft)
    _install_amalgamate_answer(terms, ir, saas, spread_c)
    _coalesce_augment_into_modify(terms)
    used = {u for t in terms for u in t.get("usages") or []}
    leftover = [u for u in _AI_USAGES if u not in used]
    if "amalgamate" not in used:
        leftover.append("amalgamate")
    if "combine_mix" not in used:
        leftover.append("combine_mix")
    if leftover:
        terms.append(
            {
                "usages": leftover,
                "forms": _with_saas(list(_DEFAULT_FORMS), saas),
                "result": "NODEF",
            }
        )
    return terms


def _terms_waiver_independent(
    ir: LegalIR, use_forms: list[str]
) -> list[dict[str, Any]]:
    """CC0/PDDL：核心活动 independent + relicense true；未讨论的 AI 活动 NODEF。"""
    usages = ["use", "copy", "modify", "combine", "amalgamate"]
    if (
        "data" in resolved_work_types(ir.grant.work_signals, ir.cues)
        or (ir.embed and ir.embed.mentioned)
    ):
        usages.append("embed")
    terms = [
        _term(
            usages,
            use_forms,
            "independent",
            relicense=True,
            copyleft=False,
            restrictions=[],
            keywords=["reproduce", "adapt", "reuse"],
        )
    ]
    used = set(usages)
    leftover = [u for u in _AI_USAGES if u not in used]
    if leftover:
        terms.append(
            {
                "usages": leftover,
                "forms": use_forms,
                "result": "NODEF",
            }
        )
    return terms


def _modify_terms_for_scope(
    ir: LegalIR,
    layer_b: list[str],
    layer_c: list[str],
    saas: bool,
    compat: list[str],
    allow_modification: bool,
    aup_pass: bool,
) -> list[dict[str, Any]]:
    """仅源码档位且两种形态换证不同时，modify 拆成两行。"""
    base = _modify_term(
        ir.modify,  # type: ignore[arg-type]
        ir.evidence,
        layer_b,
        layer_c,
        saas,
        compat,
        augment=ir.augment,
        allow_modification=allow_modification,
        aup_pass=aup_pass,
        different_terms=ir.cues.different_terms_relicense,
    )
    cues = ir.cues
    forms = list(base.get("forms") or [])
    if (
        cues.same_license_scope != "source_only"
        or "raw" not in forms
        or "binary" not in forms
        or not (cues.binary_relicense_free or cues.binary_relicense_conditional)
        or base.get("result") == "NODEF"
    ):
        if cues.same_license_scope == "source_only" and cues.source_same_by_attaching:
            base["copyleft"] = False
        elif (
            cues.same_license_scope in {"whole", "file_all_forms"}
            and base.get("result") != "NODEF"
        ):
            base["copyleft"] = True
            if not _open_compat(ir):
                base["relicense"] = False
        return [base]
    raw_forms = [item for item in forms if item != "binary"]
    bin_forms = [item for item in forms if item != "raw"]
    raw = dict(base)
    binary = dict(base)
    raw["forms"] = raw_forms or ["raw"]
    binary["forms"] = bin_forms or ["binary"]
    raw["relicense"] = "conditional" if _open_compat(ir) and not cues.source_same_by_attaching else False
    raw["copyleft"] = not cues.source_same_by_attaching and bool(base.get("copyleft") or _scope_copyleft(ir))
    binary["copyleft"] = False
    binary["relicense"] = True if cues.binary_relicense_free else "conditional"
    return [raw, binary]


def _modify_term(
    modify: ModifyFacts,
    evidence: list[EvidenceItem],
    layer_b: Optional[list[str]] = None,
    layer_c: Optional[list[str]] = None,
    saas: bool = False,
    compat: Optional[list[str]] = None,
    augment: Optional[AugmentFacts] = None,
    allow_modification: bool = True,
    aup_pass: bool = False,
    different_terms: bool = False,
) -> dict[str, Any]:
    if modify.bound == "bound":
        remix = bool(
            augment
            and augment.mentioned
            and augment.bound == "bound"
            and allow_modification
        )
        # BY/SA 同形：remix → modification；SA 义务在 copyleft/relicense，不改 result。
        result = "modification" if remix else "derivative"
    elif modify.bound == "not_bound":
        result = "independent"
    else:
        result = "NODEF"
    forms = _with_saas(modify.forms or list(_DEFAULT_FORMS), saas)
    if result == "NODEF":
        return {
            "usages": ["modify"],
            "forms": forms,
            "result": "NODEF",
        }
    copyleft = modify.same_license == "yes"
    relicense = _adapted_relicense(
        copyleft, modify.relicense, compat, aup_pass, different_terms
    )
    extra = [
        x
        for x in _norm_restrictions(modify.restrictions)
        if x in {"include_use_restriction", "include_runtime_restriction"}
    ]
    return _term(
        ["modify"],
        forms,
        result,
        relicense=relicense,
        copyleft=copyleft,
        restrictions=_dedupe((layer_b or []) + (layer_c or []) + extra),
        keywords=_keywords(evidence, modify.evidence_ids, topics=("modify_adapt",)),
    )


def _strip_usage(terms: list[dict[str, Any]], usage: str) -> None:
    """从各行拿掉某个用法。该行只剩这一个用法时删行。"""
    kept: list[dict[str, Any]] = []
    for row in terms:
        usages = [item for item in (row.get("usages") or []) if item != usage]
        if usages == list(row.get("usages") or []):
            kept.append(row)
            continue
        if not usages:
            continue
        cloned = dict(row)
        cloned["usages"] = usages
        kept.append(cloned)
    terms[:] = kept


def _install_amalgamate_answer(
    terms: list[dict[str, Any]],
    ir: LegalIR,
    saas: bool,
    spread_c: list[str],
) -> None:
    """融合题的答案写行。没讨论则撤掉套话克隆，留回未定义行。"""
    facts = ir.amalgamate
    if facts is None:
        return
    if not facts.mentioned:
        if ir.cues.amalgamate_follows_modify and not facts.saw_follow_sentence:
            return
        _strip_usage(terms, "amalgamate")
        return
    modify = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("result") in {"derivative", "modification"}
        ),
        None,
    )
    bound = facts.bound
    if bound not in {"bound", "not_bound"}:
        if facts.same_as_modify == "yes" and modify is not None:
            bound = "bound" if modify.get("result") != "independent" else "not_bound"
        else:
            return
    if bound == "bound":
        result = "derivative"
    else:
        result = "independent"
    inherit = facts.same_as_modify == "yes" or (
        facts.relicense in {"", "not_mentioned", "n/a"}
        and ir.cues.amalgamate_follows_modify
    )
    stated = facts.relicense
    if (
        not facts.relicense_settled
        and stated in {"no", "false"}
        and facts.same_as_modify == "yes"
        and modify is not None
    ):
        relicense = modify.get("relicense", False)
    elif stated in {"", "not_mentioned", "n/a"} and inherit and modify is not None:
        relicense = modify.get("relicense", False)
    elif stated in {"", "not_mentioned", "n/a"}:
        relicense = False
    else:
        relicense = _adapted_relicense(
            facts.same_as_modify == "yes" and bool(modify and modify.get("copyleft")),
            stated,
            aup_pass=bool(
                ir.restrict and ir.restrict.pass_on == "yes" and stated in {"yes", "true"}
            ),
            different_terms=False,
        )
    if relicense is False and ir.named.adopted:
        relicense = "conditional"
    if facts.restrictions_stated:
        restrictions = list(facts.restrictions)
    elif inherit and modify is not None:
        restrictions = [
            item
            for item in (modify.get("restrictions") or [])
            if item in {"include_notice", "include_license", "state_changes"}
        ]
    else:
        restrictions = []
    for item in spread_c:
        if item not in restrictions:
            restrictions.append(item)
    copyleft = bool(modify and modify.get("copyleft")) and facts.same_as_modify == "yes"
    _strip_usage(terms, "amalgamate")
    terms.append(
        _term(
            ["amalgamate"],
            _with_saas(list(_DEFAULT_FORMS), saas),
            result,
            relicense=relicense,
            copyleft=copyleft,
            restrictions=restrictions,
        )
    )


def _clone_amalgamate_from_modify(
    terms: list[dict[str, Any]],
    ir: LegalIR,
    saas: bool,
    work_copyleft: bool,
) -> None:
    """修改版与衍生作品整体同一句授权时，amalgamate 继承 modify。

    已有 absorbed / db_assimilate 行、copyleft、或 modify 不是改编结果时不补。
    ShareAlike/copyleft 不走克隆（避免 Collection 被当成整包 SA）。
    """
    if work_copyleft or not ir.cues.amalgamate_follows_modify:
        return
    if any(
        "amalgamate" in (row.get("usages") or []) and row.get("result") != "NODEF"
        for row in terms
    ):
        return
    source = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("result") in {"derivative", "modification"}
        ),
        None,
    )
    if source is None:
        return
    clone = dict(source)
    clone["usages"] = ["amalgamate"]
    if source.get("keywords"):
        clone["keywords"] = list(source["keywords"])
    clone["forms"] = _with_saas(list(source.get("forms") or []), saas)
    terms.append(clone)


def _align_amalgamate_with_modify(
    terms: list[dict[str, Any]],
    ir: LegalIR,
    work_copyleft: bool,
) -> None:
    """非 copyleft：amalgamate 与 modify 同句授权时，对齐 relicense/restrictions（可合并计分）。"""
    if work_copyleft or not ir.cues.amalgamate_follows_modify:
        return
    source = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("result") in {"derivative", "modification"}
            and not row.get("copyleft")
        ),
        None,
    )
    if source is None:
        return
    for row in terms:
        if row.get("result") in (None, "NODEF"):
            continue
        if "amalgamate" not in (row.get("usages") or []):
            continue
        if row.get("copyleft"):
            continue
        if row.get("result") not in {"derivative", "modification"}:
            continue
        row["relicense"] = source.get("relicense")
        row["restrictions"] = list(source.get("restrictions") or [])
        row["result"] = source.get("result")


def _has_distill(evidence: list[EvidenceItem]) -> bool:
    return any(e.topic == "output_distill" for e in evidence)


def _improve_other(ir: LegalIR) -> str:
    if ir.output is None:
        return "not_mentioned"
    return ir.output.improve_other or "not_mentioned"


def _ai_terms(
    ir: LegalIR,
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
    compat: list[str],
    aup_pass: bool = False,
    allow_content_results: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ban_distill = _improve_other(ir) == "no"
    distill_def = _has_distill(ir.evidence)
    train = ir.train
    train_bound = bool(train and train.mentioned and train.bound == "bound")
    distill_on_train = distill_def and train_bound and not ban_distill
    distill_mentioned = distill_def or _improve_other(ir) in {"yes", "no"}

    if ir.output is not None and ir.output.mentioned:
        rows.extend(
            _output_terms(
                ir.output,
                ir.evidence,
                forms,
                layer_c,
                split_distill=ban_distill,
                bundle_distill=(
                    distill_mentioned and not ban_distill and not distill_on_train
                ),
                aup_pass=aup_pass,
            )
        )
    elif layer_c:
        # 仅有 C 层义务、未定案输出是否受约束：不因全局 AUP 改写 generate.relicense。
        rows.append(
            _term(
                ["generate"],
                forms,
                "duplicate",
                relicense=False,
                copyleft=False,
                restrictions=layer_c,
            )
        )
    if ban_distill and not any("distill" in (r.get("usages") or []) for r in rows):
        rows.append(
            _term(
                ["distill"],
                forms,
                "derivative",
                relicense=_copyleft_relicense(compat),
                copyleft=False,
                restrictions=[],
                keywords=_keywords(
                    ir.evidence, [], topics=("output_distill", "output_generate")
                ),
            )
        )
    train_row = _train_term(
        train,
        ir.evidence,
        forms,
        layer_b,
        layer_c,
        compat,
        include_distill=distill_on_train,
        aup_pass=aup_pass,
    )
    if train_row:
        rows.append(train_row)
    embed_row = _embed_term(
        ir.embed,
        ir.evidence,
        forms,
        layer_b,
        layer_c,
        ir.grant,
        allow_translation=allow_content_results,
    )
    if embed_row:
        rows.append(embed_row)
    aug_row = _augment_term(
        ir.augment,
        ir.evidence,
        forms,
        layer_b,
        layer_c,
        allow_modification=allow_content_results,
    )
    if aug_row:
        rows.append(aug_row)
    stat_row = _stat_term(ir.stat, ir.evidence, forms, layer_b, layer_c)
    if stat_row:
        rows.append(stat_row)
    return rows


def _apply_distill_named_restriction(
    terms: list[dict[str, Any]], ir: LegalIR
) -> list[dict[str, Any]]:
    """蒸馏仍是 NODEF，且限制句点名了衍生或产出时，单独带上已成立的使用限制。"""
    if not ir.cues.distill_named_restriction:
        return terms
    use = bool(ir.restrict and ir.restrict.mentioned)
    runtime = bool(ir.restrict and ir.restrict.runtime == "yes")
    for row in terms:
        items = row.get("restrictions") or []
        if "include_use_restriction" in items:
            use = True
        if "include_runtime_restriction" in items:
            runtime = True
    if not use:
        return terms
    restrictions = ["include_use_restriction"]
    if runtime:
        restrictions.append("include_runtime_restriction")
    out: list[dict[str, Any]] = []
    for row in terms:
        usages = list(row.get("usages") or [])
        if "distill" not in usages or row.get("result") not in (None, "", "NODEF"):
            out.append(row)
            continue
        kept = [item for item in usages if item != "distill"]
        if kept:
            kept_row = dict(row)
            kept_row["usages"] = kept
            out.append(kept_row)
        distill_row = dict(row)
        distill_row["usages"] = ["distill"]
        distill_row["restrictions"] = list(restrictions)
        out.append(distill_row)
    return out


def _apply_amalgamate_state_changes(
    terms: list[dict[str, Any]], ir: LegalIR
) -> list[dict[str, Any]]:
    """合并只补声明变更。不改 result、relicense、copyleft，也不写到其余用法上。"""
    facts = ir.amalgamate
    if facts is not None and (not facts.mentioned or facts.restrictions_stated):
        return terms
    if not ir.cues.derivative_must_state_changes:
        return terms
    modify_has = any(
        "modify" in (row.get("usages") or [])
        and "state_changes" in (row.get("restrictions") or [])
        for row in terms
    )
    if not modify_has:
        return terms
    if any(
        "amalgamate" in (row.get("usages") or [])
        and "state_changes" in (row.get("restrictions") or [])
        for row in terms
    ):
        return terms
    out: list[dict[str, Any]] = []
    for row in terms:
        usages = list(row.get("usages") or [])
        if "amalgamate" not in usages:
            out.append(row)
            continue
        kept = [item for item in usages if item != "amalgamate"]
        if kept:
            kept_row = dict(row)
            kept_row["usages"] = kept
            out.append(kept_row)
        amalgam = dict(row)
        amalgam["usages"] = ["amalgamate"]
        amalgam["restrictions"] = ["state_changes"]
        out.append(amalgam)
    return out


def _apply_embed_inherits_modify(
    terms: list[dict[str, Any]], ir: LegalIR
) -> list[dict[str, Any]]:
    """嵌入整段继承修改。已经是 not_bound 的不覆盖。"""
    if not ir.cues.embed_inherits_modify:
        return terms
    modify = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("result") not in (None, "", "NODEF")
        ),
        None,
    )
    if modify is None:
        return terms
    out: list[dict[str, Any]] = []
    for row in terms:
        usages = list(row.get("usages") or [])
        if "embed" not in usages or row.get("result") == "independent":
            out.append(row)
            continue
        kept = [item for item in usages if item != "embed"]
        if kept:
            kept_row = dict(row)
            kept_row["usages"] = kept
            out.append(kept_row)
        out.append(
            {
                "usages": ["embed"],
                "forms": list(row.get("forms") or []),
                "result": modify.get("result"),
                "restrictions": list(modify.get("restrictions") or []),
                "relicense": modify.get("relicense", False),
                "copyleft": bool(modify.get("copyleft")),
            }
        )
    return out


def _apply_other_kind_exception(
    terms: list[dict[str, Any]], ir: LegalIR
) -> list[dict[str, Any]]:
    """例外点名另一类作品时写入 combine_mix。同类型组合保持修改行的 copyleft。"""
    if not ir.cues.other_kind_exception:
        return terms
    modify = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("copyleft")
            and row.get("result") not in (None, "", "NODEF")
        ),
        None,
    )
    if modify is None:
        return terms
    out: list[dict[str, Any]] = []
    saw_mix = False
    for row in terms:
        usages = list(row.get("usages") or [])
        if "combine_mix" in usages and "combine" not in usages:
            saw_mix = True
            out.append(row)
            continue
        if "combine" not in usages or "modify" in usages or "amalgamate" in usages:
            out.append(row)
            continue
        if (
            row.get("copyleft")
            and row.get("result") in {"derivative", "modification"}
            and "combine_mix" not in usages
        ):
            out.append(row)
            continue
        kept = [item for item in usages if item != "combine"]
        if kept:
            kept_row = dict(row)
            kept_row["usages"] = kept
            if "combine_mix" in kept:
                saw_mix = True
            out.append(kept_row)
        out.append(
            {
                "usages": ["combine"],
                "forms": list(row.get("forms") or ["raw", "binary"]),
                "result": "derivative",
                "restrictions": list(modify.get("restrictions") or []),
                "relicense": modify.get("relicense", False),
                "copyleft": True,
            }
        )
    if not saw_mix:
        out.append(
            {
                "usages": ["combine_mix"],
                "forms": ["raw", "binary"],
                "result": "independent",
                "restrictions": [],
                "relicense": True,
                "copyleft": False,
            }
        )
    return out


def _apply_your_derivative_relicense(
    terms: list[dict[str, Any]], ir: LegalIR
) -> list[dict[str, Any]]:
    """你做的衍生可以另附条款时，未定义的合并写成衍生并可换证。

    融合题已经回答过时，不再用这句改写。
    """
    if ir.amalgamate is not None:
        return terms
    if not ir.cues.your_derivative_may_relicense:
        return terms
    modify = next(
        (
            row
            for row in terms
            if "modify" in (row.get("usages") or [])
            and row.get("result") not in (None, "", "NODEF")
        ),
        None,
    )
    if modify is None:
        return terms
    out: list[dict[str, Any]] = []
    for row in terms:
        usages = list(row.get("usages") or [])
        if "amalgamate" not in usages or row.get("result") not in (None, "", "NODEF"):
            out.append(row)
            continue
        kept = [item for item in usages if item != "amalgamate"]
        if kept:
            kept_row = dict(row)
            kept_row["usages"] = kept
            out.append(kept_row)
        out.append(
            {
                "usages": ["amalgamate"],
                "forms": list(row.get("forms") or []),
                "result": modify.get("result"),
                "restrictions": ["include_use_restriction", "state_changes"],
                "relicense": True,
                "copyleft": False,
            }
        )
    return out


def _output_terms(
    output: OutputFacts,
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_c: Optional[list[str]],
    split_distill: bool,
    bundle_distill: bool,
    aup_pass: bool = False,
) -> list[dict[str, Any]]:
    row = _output_term(output, evidence, forms, layer_c, aup_pass=aup_pass)
    if not row:
        return []
    if split_distill:
        row["usages"] = ["generate"]
        return [row]
    if bundle_distill:
        row["usages"] = ["generate", "distill"]
        return [row]
    row["usages"] = ["generate"]
    return [row]


def _train_term(
    train: Optional[TrainFacts],
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
    compat: list[str],
    include_distill: bool,
    aup_pass: bool = False,
) -> Optional[dict[str, Any]]:
    if train is None or not train.mentioned:
        return None
    if train.bound == "bound":
        result = "derivative"
        copyleft = False
        relicense = _adapted_relicense(
            False, train.relicense, compat, aup_pass
        )
        restrictions = _dedupe(list(layer_b) + list(layer_c))
    elif train.bound == "not_bound":
        result = "independent"
        copyleft = False
        relicense = _adapted_relicense(
            False, train.relicense, compat, aup_pass
        )
        restrictions = list(layer_c)
    else:
        return None
    usages = ["train"]
    if include_distill:
        usages.append("distill")
    return _term(
        usages,
        forms,
        result,
        relicense=relicense,
        copyleft=copyleft,
        restrictions=restrictions,
        keywords=_keywords(
            evidence, train.evidence_ids, topics=("train_finetune", "output_distill")
        ),
    )


def _embed_term(
    embed: Optional[EmbedFacts],
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
    grant: GrantFacts,
    allow_translation: bool = True,
) -> Optional[dict[str, Any]]:
    if embed is None or not embed.mentioned:
        return None
    kind = embed.result_kind
    bound = embed.bound
    if bound == "translation" or kind == "translation":
        result = "translation" if allow_translation else "derivative"
        copyleft = False
        relicense = _norm_relicense(embed.relicense, default=False)
        restrictions = _dedupe(list(layer_b) + list(layer_c))
    elif bound == "bound":
        result = "derivative"
        copyleft = grant.same_license_this == "yes"
        relicense = _norm_relicense(
            embed.relicense, default=False if copyleft else True
        )
        restrictions = _dedupe(list(layer_b) + list(layer_c))
    elif bound == "not_bound":
        result = "independent"
        copyleft = False
        relicense = _norm_relicense(embed.relicense, default=True)
        restrictions = list(layer_c)
    else:
        return None
    return _term(
        ["embed"],
        forms,
        result,
        relicense=relicense,
        copyleft=copyleft,
        restrictions=restrictions,
        keywords=_keywords(evidence, embed.evidence_ids, topics=("embed_adapt",)),
    )


def _reuse_row(
    bound: str,
    relicense: str,
    usages: list[str],
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
    keywords: list[str],
    bound_result: str,
) -> Optional[dict[str, Any]]:
    if bound == "bound":
        result = bound_result
        relicense_val = _norm_relicense(relicense, default=False)
        restrictions = _dedupe(list(layer_b) + list(layer_c))
    elif bound == "not_bound":
        result = "independent"
        relicense_val = _norm_relicense(relicense, default=True)
        restrictions = list(layer_c)
    else:
        return None
    return _term(
        usages,
        forms,
        result,
        relicense=relicense_val,
        copyleft=False,
        restrictions=restrictions,
        keywords=keywords,
    )


def _augment_term(
    augment: Optional[AugmentFacts],
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
    allow_modification: bool = True,
) -> Optional[dict[str, Any]]:
    if augment is None or not augment.mentioned:
        return None
    bound_result = "modification" if allow_modification else "derivative"
    return _reuse_row(
        augment.bound,
        augment.relicense,
        ["augment"],
        forms,
        layer_b,
        layer_c,
        _keywords(evidence, augment.evidence_ids, topics=("augment_remix",)),
        bound_result,
    )


def _stat_term(
    stat: Optional[StatFacts],
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_b: list[str],
    layer_c: list[str],
) -> Optional[dict[str, Any]]:
    if stat is None or not stat.mentioned:
        return None
    if stat.bound not in {"bound", "not_bound"}:
        return None
    return _reuse_row(
        stat.bound,
        stat.relicense,
        ["stat"],
        forms,
        layer_b,
        layer_c,
        _keywords(evidence, stat.evidence_ids, topics=("stat_compute",)),
        "derivative",
    )


def _output_term(
    output: OutputFacts,
    evidence: list[EvidenceItem],
    forms: list[str],
    layer_c: Optional[list[str]] = None,
    aup_pass: bool = False,
) -> Optional[dict[str, Any]]:
    bound = output.bound
    use_rules = output.use_rules == "yes"
    if bound == "bound" or use_rules:
        result = "derivative"
        copyleft = bound == "bound"
        # 仅当输出已被定案为受使用限制（O2）且非 bound-copyleft 时，AUP 才抬 conditional。
        # 全局有 AUP ≠ 产物必然受约束。
        relicense = (
            _adapted_relicense(False, "not_mentioned", None, aup_pass)
            if aup_pass and use_rules and not copyleft
            else False
        )
        restrictions = ["include_use_restriction"] if use_rules else []
    elif bound in {"not_bound", "only_if_output_is_covered"}:
        result = "independent"
        copyleft = False
        relicense = True
        restrictions = []
    else:
        return None
    return _term(
        ["generate", "distill"],
        forms or list(_DEFAULT_FORMS),
        result,
        relicense=relicense,
        copyleft=copyleft,
        restrictions=_dedupe(restrictions + (layer_c or [])),
        keywords=_keywords(evidence, output.evidence_ids, topics=("output_generate",)),
    )


def _combine_terms(
    combine: CombineFacts,
    evidence: list[EvidenceItem],
    compat: Optional[list[str]] = None,
    layer_b: Optional[list[str]] = None,
    layer_c: Optional[list[str]] = None,
    saas: bool = False,
    work_copyleft: bool = False,
    aup_pass: bool = False,
    default_independent: bool = False,
) -> list[dict[str, Any]]:
    if not combine.mentioned or not combine.situations:
        rows = [
            {
                "usages": ["combine"],
                "forms": _with_saas(list(_DEFAULT_FORMS), saas),
                "result": "NODEF",
            }
        ]
    else:
        rows = []
        c2 = combine.other_work_must_same_license
        for sit in combine.situations:
            rows.extend(
                _situation_rows(
                    sit,
                    c2,
                    evidence,
                    compat or [],
                    layer_b or [],
                    layer_c or [],
                    saas,
                    work_copyleft=work_copyleft,
                    aup_pass=aup_pass,
                )
            )
        rows = _merge_same_semantics(_coalesce_combine(rows))
    return _fill_missing_combine(
        rows,
        combine,
        aup_pass,
        layer_b or [],
        layer_c or [],
        saas,
        default_independent=default_independent,
    )


def _combine_carveout(combine: CombineFacts) -> bool:
    """原文明确把某种组合排除出本许可（聚合，或另一作品不必同许可）。"""
    c2 = combine.other_work_must_same_license
    for sit in combine.situations or []:
        if sit.id == "agg":
            return True
        if sit.id == "other_uses" and c2 == "no":
            return True
    return False


def _defined_combine_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if "combine" in (r.get("usages") or [])
        and r.get("result") not in (None, "NODEF")
    ]


def _drop_nodef_combine(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if not (
            (r.get("usages") or []) == ["combine"]
            and r.get("result") in (None, "NODEF")
        )
    ]


def _fill_missing_combine(
    rows: list[dict[str, Any]],
    combine: CombineFacts,
    aup_pass: bool,
    layer_b: list[str],
    layer_c: list[str],
    saas: bool,
    default_independent: bool = False,
) -> list[dict[str, Any]]:
    """已有明确 combine 行则不动。否则：AUP 传递 → derivative；默认可再分发 → independent。"""
    if _defined_combine_rows(rows):
        return rows
    if aup_pass and not _combine_carveout(combine):
        kept = _drop_nodef_combine(rows)
        kept.append(
            _term(
                ["combine"],
                _with_saas(["raw", "binary"], saas),
                "derivative",
                relicense="conditional",
                copyleft=False,
                restrictions=_dedupe(list(layer_b) + list(layer_c)),
            )
        )
        return kept
    if default_independent:
        kept = _drop_nodef_combine(rows)
        kept.append(
            _term(
                ["combine"],
                _with_saas(["raw", "binary"], saas),
                "independent",
                relicense=True,
                copyleft=False,
                restrictions=[],
            )
        )
        return kept
    return rows


def _independent_combine_forms(sit: CombineSituation) -> list[str]:
    forms = list(sit.forms or [])
    return _dedupe(forms + ["raw", "binary"]) if forms else ["raw", "binary"]


def _default_forms_for(sid: str) -> list[str]:
    if sid == "agg":
        return ["binary"]
    if sid == "other_uses":
        return ["raw"]
    if sid == "absorbed":
        return ["raw", "binary"]
    if sid == "db_assimilate":
        return ["raw", "binary"]
    return list(_DEFAULT_FORMS)


def _copyleft_relicense(compat: list[str]) -> Any:
    """同许可义务下默认可换兼容许可；compat 名单有无只影响名单字段，不改成死 false。"""
    return "conditional"


def _absorbed_forms(sit: CombineSituation) -> list[str]:
    forms = sit.forms or _default_forms_for("absorbed")
    if "saas" in forms:
        return _dedupe(forms)
    if "binary" not in forms:
        return _dedupe(list(forms) + ["binary"])
    return _dedupe(forms)


def _situation_rows(
    sit: CombineSituation,
    c2: str,
    evidence: list[EvidenceItem],
    compat: Optional[list[str]] = None,
    layer_b: Optional[list[str]] = None,
    layer_c: Optional[list[str]] = None,
    saas: bool = False,
    work_copyleft: bool = False,
    aup_pass: bool = False,
) -> list[dict[str, Any]]:
    forms = sit.forms or _default_forms_for(sit.id)
    keywords = _keywords(evidence, sit.evidence_ids)
    deriv_restr = _dedupe((layer_b or []) + (layer_c or []))
    indep_restr = list(layer_c or [])
    if sit.id == "agg":
        return [
            _term(
                ["combine"],
                _with_saas(forms, saas),
                "independent",
                relicense=_norm_relicense(sit.relicense, default=True),
                copyleft=False,
                restrictions=indep_restr,
                keywords=keywords,
            )
        ]
    if sit.id == "other_uses":
        # AUP 传递 ≠ 另一作品必须同许可；bound/假 C2 仍出 derivative+conditional。
        aup_blocks_infect = aup_pass and not work_copyleft
        if aup_blocks_infect and (c2 == "yes" or sit.bound == "bound"):
            return [
                _term(
                    ["combine"],
                    _with_saas(forms, saas),
                    "derivative",
                    relicense=_adapted_relicense(
                        False, sit.relicense, compat or [], aup_pass=True
                    ),
                    copyleft=False,
                    restrictions=deriv_restr,
                    keywords=keywords,
                )
            ]
        if c2 == "yes":
            return [
                _term(
                    ["combine"],
                    forms,
                    "derivative",
                    relicense=_copyleft_relicense(compat or []),
                    copyleft=True,
                    restrictions=deriv_restr,
                    keywords=keywords,
                )
            ]
        forms = _independent_combine_forms(sit)
        return [
            _term(
                ["combine"],
                _with_saas(forms, saas),
                "independent",
                relicense=True,
                copyleft=False,
                restrictions=indep_restr,
                keywords=keywords,
            )
        ]
    if sit.id == "absorbed":
        # 纯 AUP + 模型乱答 C2：不 infect；真 work_copyleft 仍可传染。
        infect = (c2 == "yes" or work_copyleft) and not (
            aup_pass and not work_copyleft
        )
        return [
            _term(
                ["amalgamate"],
                _with_saas(_absorbed_forms(sit), saas),
                "derivative",
                relicense=(
                    _copyleft_relicense(compat or [])
                    if infect
                    else _adapted_relicense(
                        False, sit.relicense, compat or [], aup_pass
                    )
                ),
                copyleft=infect,
                restrictions=deriv_restr,
                keywords=keywords,
            )
        ]
    if sit.id == "db_assimilate":
        infect = work_copyleft
        relic = (
            _copyleft_relicense(compat or [])
            if infect
            else _norm_relicense(sit.relicense, default="conditional")
        )
        return [
            _term(
                ["amalgamate"],
                _with_saas(forms or list(_DEFAULT_FORMS), saas),
                "derivative",
                relicense=relic,
                copyleft=infect,
                restrictions=deriv_restr,
                keywords=keywords,
            )
        ]
    if sit.id == "mixed_type":
        if sit.bound == "bound":
            return [
                _term(
                    ["combine_mix"],
                    _with_saas(forms, saas),
                    "derivative",
                    relicense=_norm_relicense(sit.relicense, default=False),
                    copyleft=False,
                    restrictions=deriv_restr,
                    keywords=keywords,
                )
            ]
        return [
            _term(
                ["combine_mix"],
                _with_saas(forms, saas),
                "independent",
                relicense=_norm_relicense(sit.relicense, default=True),
                copyleft=False,
                restrictions=indep_restr,
                keywords=keywords,
            )
        ]
    return []


def _coalesce_augment_into_modify(terms: list[dict[str, Any]]) -> None:
    """Adapted Material：同形态的 augment 并入 modify（不碰 combine 行）。

    BY/SA：result 均为 modification 时可合；SA 时以 modify 的 copyleft/relicense 为准。
    """
    modify_i: Optional[int] = None
    augment_i: Optional[int] = None
    for i, row in enumerate(terms):
        if row.get("result") in (None, "NODEF"):
            continue
        us = row.get("usages") or []
        if us == ["modify"]:
            modify_i = i
        elif us == ["augment"]:
            augment_i = i
    if modify_i is None or augment_i is None:
        return
    mod = terms[modify_i]
    aug = terms[augment_i]
    if mod.get("result") not in {"modification", "derivative"}:
        return
    if aug.get("result") not in {"modification", "derivative"}:
        return
    # 形态不一致（一端 modification 一端 derivative）时，优先保留 modification（内容改编）。
    if mod.get("result") != aug.get("result"):
        if "modification" in {mod.get("result"), aug.get("result")}:
            mod["result"] = "modification"
        else:
            return
    if mod.get("relicense") != aug.get("relicense") or bool(mod.get("copyleft")) != bool(
        aug.get("copyleft")
    ):
        # 以 modify（含 ShareAlike）义务为准对齐 remix 行
        aug = dict(aug)
        aug["relicense"] = mod.get("relicense")
        aug["copyleft"] = mod.get("copyleft")
        aug["restrictions"] = list(mod.get("restrictions") or [])
    usages = {"augment", "modify"}
    mod["usages"] = [u for u in ("augment", "modify") if u in usages]
    mod["restrictions"] = _dedupe(
        list(mod.get("restrictions") or []) + list(aug.get("restrictions") or [])
    )
    mod["keywords"] = _dedupe(
        list(mod.get("keywords") or []) + list(aug.get("keywords") or [])
    )
    if aug.get("result") == "modification":
        mod["result"] = "modification"
    del terms[augment_i]


def _coalesce_combine(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一 usages+forms 合并；result 取更严（derivative > independent > NODEF）。"""
    rank = {
        "NODEF": 0,
        "independent": 1,
        "duplicate": 1,
        "derivative": 2,
        "modification": 2,
        "translation": 2,
    }
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    order: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for row in rows:
        key = (tuple(row.get("usages") or []), tuple(row.get("forms") or []))
        if key not in grouped:
            grouped[key] = row
            order.append(key)
            continue
        cur = grouped[key]
        if rank.get(row.get("result"), 0) > rank.get(cur.get("result"), 0):
            merged = dict(row)
            merged["copyleft"] = bool(row.get("copyleft") or cur.get("copyleft"))
            merged["keywords"] = _dedupe((cur.get("keywords") or []) + (row.get("keywords") or []))
            grouped[key] = merged
        else:
            cur["copyleft"] = bool(cur.get("copyleft") or row.get("copyleft"))
            cur["keywords"] = _dedupe((cur.get("keywords") or []) + (row.get("keywords") or []))
    return [grouped[k] for k in order]


def _merge_same_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """result/copyleft/relicense 相同的 combine 行合并 forms（LGPL 两形态同独立）。"""
    merged: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for row in rows:
        key = (
            tuple(row.get("usages") or []),
            row.get("result"),
            row.get("copyleft"),
            row.get("relicense"),
        )
        if key not in merged:
            merged[key] = dict(row)
            merged[key]["forms"] = list(row.get("forms") or [])
            order.append(key)
            continue
        cur = merged[key]
        cur["forms"] = _dedupe((cur.get("forms") or []) + (row.get("forms") or []))
        cur["keywords"] = _dedupe((cur.get("keywords") or []) + (row.get("keywords") or []))
        cur["restrictions"] = _dedupe((cur.get("restrictions") or []) + (row.get("restrictions") or []))
    return [merged[k] for k in order]


def _term(
    usages: list[str],
    forms: list[str],
    result: str,
    relicense: Any = False,
    copyleft: bool = False,
    restrictions: Optional[list[str]] = None,
    keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "usages": usages,
        "forms": _dedupe(forms),
        "result": result,
        "restrictions": restrictions or [],
        "relicense": relicense,
        "copyleft": copyleft,
    }
    if keywords:
        row["keywords"] = keywords
    return row


def _norm_relicense(value: str, default: Any) -> Any:
    if value in {"true", "yes"}:
        return True
    if value in {"false", "no"}:
        return False
    if value == "conditional":
        return "conditional"
    return default


def _norm_restrictions(values: list[str]) -> list[str]:
    allowed = {
        "include_notice",
        "include_license",
        "state_changes",
        "include_use_restriction",
        "include_runtime_restriction",
        "keep_notice",
        "keep_license_text",
        "state_changes",
        "use_rules_follow",
        "runtime_rules_follow",
    }
    mapped = {
        "keep_notice": "include_notice",
        "keep_license_text": "include_license",
        "use_rules_follow": "include_use_restriction",
        "runtime_rules_follow": "include_runtime_restriction",
    }
    out: list[str] = []
    for item in values or []:
        if item in {"none", "not_mentioned"}:
            continue
        out.append(mapped.get(item, item))
    return _dedupe([x for x in out if x in {
        "include_notice",
        "include_license",
        "state_changes",
        "include_use_restriction",
        "include_runtime_restriction",
    }])


def _keywords(
    evidence: list[EvidenceItem],
    ids: list[str],
    limit: int = 8,
    topics: Optional[tuple[str, ...]] = None,
) -> list[str]:
    idset = set(ids or [])
    items = [e for e in evidence if e.id in idset] if idset else []
    if not items:
        if topics:
            items = [e for e in evidence if e.topic in topics][:6]
        else:
            items = [e for e in evidence if e.topic.startswith("combine")][:6]
    words: list[str] = []
    seen: set[str] = set()
    for e in items:
        cue = (e.cue or "").strip()
        if not cue:
            continue
        key = cue.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(cue)
        if len(words) >= limit:
            return words
    return words


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
