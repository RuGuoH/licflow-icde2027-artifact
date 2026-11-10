"""用已经送进换证题的句子核对模型答案。不按许可名称分支。"""
from __future__ import annotations

import re

_MUST_STAY_RE = re.compile(
    r"licensed as a whole.{0,160}under the terms of this (?:License|Agreement)|"
    r"must license the entire work, as a whole, under this|"
    r"no permission to license the work in any other way|"
    r"modified work as a whole|"
    r"shall be licensed under this|"
    r"only under this (?:license|licence|Agreement)|"
    r"must not be distributed under any other (?:license|licence)",
    re.IGNORECASE,
)
_NEW_TERMS_RE = re.compile(
    r"must not conflict|"
    r"conflict with|"
    r"do(?:es)? not contradict|"
    r"respecting paragraph|"
    r"such licen[cs]e terms (?:limit|require)|"
    r"use[- ]based restrictions.{0,220}enforceable provision|"
    r"use restrictions.{0,220}enforceable provision|"
    r"compatible licen[cs]e|"
    r"Secondary Licen[cs]e|"
    r"licen[cs]e of your choice that does not contradict",
    re.IGNORECASE,
)
_NON_SUBLICENSE_RE = re.compile(r"non[-\s]?sublicensable", re.IGNORECASE)
_SUBLICENSABLE_RE = re.compile(r"\bsublicensable\b", re.IGNORECASE)
_WITHOUT_RESTRICTION_RE = re.compile(r"\bwithout restriction\b", re.IGNORECASE)
_DO_NOT_DISTRIBUTE_RE = re.compile(
    r"\b(?:do|does|must|shall|may)\s+not\s+(?:distribut|redistribut)\w*",
    re.IGNORECASE,
)
# 同一句里授予修改、改编或再许可，或分发修改后的版本。逐字副本和链接结果不算。
_UNRESTRICTED_MODIFY_RE = re.compile(
    r"\bdeal in\b"
    r"|\bsublicen[cs]e\b"
    r"|\b(?:modify|adapt|alter)\b"
    r"|\b(?:distribut|redistribut)\w*(?:\s+\w+){0,12}\s+"
    r"(?:modif(?:ied|ication)|adapt(?:ed|ation)|alter(?:ed|ation)|changed)\b",
    re.IGNORECASE,
)
_OTHER_TERMS_YES_RE = re.compile(
    r"may provide additional or different licen[cs]e terms",
    re.IGNORECASE,
)
_OTHERWISE_RE = re.compile(r"otherwise complies", re.IGNORECASE)


def _without_restriction_yes(sentence: str) -> bool:
    if not _WITHOUT_RESTRICTION_RE.search(sentence or ""):
        return False
    if _DO_NOT_DISTRIBUTE_RE.search(sentence):
        return False
    return _UNRESTRICTED_MODIFY_RE.search(sentence) is not None


def _grant_yes(text: str) -> bool:
    raw = text or ""
    cleaned = _NON_SUBLICENSE_RE.sub(" ", raw)
    if _SUBLICENSABLE_RE.search(cleaned):
        return True
    for piece in re.split(r"[\n]+|(?<=[.!?;])\s+", raw):
        if _without_restriction_yes(piece):
            return True
    return False


def _other_terms_yes(text: str) -> bool:
    return bool(_OTHER_TERMS_YES_RE.search(text or "") and _OTHERWISE_RE.search(text or ""))


def _flat(unit: str) -> str:
    return re.sub(r"\s+", " ", unit or "").strip()


def focus_units(text: str, *, limit: int = 12) -> list[str]:
    """全文里直接决定换证的句子。问卷配额没抽到时，仍送进这一题。"""
    from scripts.license_modeling_qa.extract import split_units

    found: list[str] = []
    for _start, _end, unit in split_units(text or ""):
        flat = _flat(unit)
        if not flat:
            continue
        if (
            _MUST_STAY_RE.search(flat)
            or _NEW_TERMS_RE.search(flat)
            or _other_terms_yes(flat)
            or _grant_yes(flat)
        ):
            found.append(flat[:1600])
        if len(found) >= limit:
            break
    return found


def relicense_evidence_packets(
    license_text: str, evidence: list
) -> tuple[list, list]:
    """修改、训练两道换证题的引文。决定句补进配额，引文用完整句子。"""
    import re
    from dataclasses import replace

    from .extract import (
        evidence_for_topics,
        is_short_fulltext,
        make_fulltext_evidence,
        supplement_relicense_evidence,
    )
    from .gazetteer import DISTILL_TOPICS, MODIFY_TOPICS, TRAIN_TOPICS
    from .types import EvidenceItem

    def expand(items: list) -> list:
        expanded = []
        for item in items:
            if item.start is None or item.end is None:
                expanded.append(item)
                continue
            quote = re.sub(r"\s+", " ", license_text[item.start : item.end]).strip()[:1600]
            expanded.append(replace(item, quote=quote))
        return expanded

    def with_focus(items: list) -> list:
        extra = [
            EvidenceItem(
                id="focus%d" % index,
                topic="modify_adapt",
                quote=unit[:1600],
                cue="relicense_focus",
            )
            for index, unit in enumerate(focus_units(license_text))
        ]
        seen: set[str] = set()
        merged = []
        for item in extra + list(items):
            key = (item.quote or "")[:120]
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged[:24]

    if is_short_fulltext(license_text):
        full = with_focus([make_fulltext_evidence(license_text)])
        return full, full
    modify_ev = evidence_for_topics(evidence, MODIFY_TOPICS)
    extra = supplement_relicense_evidence(evidence, modify_ev)
    modify_ev = with_focus(expand(list(modify_ev) + extra))
    train_ev = evidence_for_topics(
        evidence, list(TRAIN_TOPICS | DISTILL_TOPICS), limit=10
    )
    seen_ids = {item.id for item in train_ev}
    train_ev = list(train_ev) + [item for item in extra if item.id not in seen_ids]
    train_ev = with_focus(expand(train_ev) + modify_ev)
    return modify_ev, train_ev


def settle_relicense(sentences: list[str], stated: str | None) -> str | None:
    """必须用本许可优先，其次新条款受限，再其次可再许可。都没有则保留模型答案。"""
    blob = "\n".join(sentences)
    if _MUST_STAY_RE.search(blob):
        return "no"
    if _NEW_TERMS_RE.search(blob):
        return "conditional"
    if any(_other_terms_yes(sentence) for sentence in sentences):
        return "yes"
    if _grant_yes(blob):
        return "yes"
    if stated == "conditional":
        return "conditional"
    if stated == "yes" and not _NON_SUBLICENSE_RE.search(blob):
        return "yes"
    return None
