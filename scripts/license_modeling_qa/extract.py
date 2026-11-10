"""词表预抽 + 缺证再补。无 LLM。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .gazetteer import COMBINE_TOPICS, GAP_TOPICS, PATTERNS
from .ir import IdentityFacts, TextCues
from .types import EvidenceItem

_COMBINE_SALIENT = (
    (re.compile(r"terms of your choice", re.IGNORECASE), 100),
    (re.compile(r"license the entire work", re.IGNORECASE), 100),
    (re.compile(r"mere aggregation|called an ['\"]?aggregate['\"]?", re.IGNORECASE), 80),
    (
        re.compile(
            r"does not (?:bring the other work|cause this license)",
            re.IGNORECASE,
        ),
        80,
    ),
    (re.compile(r"not based on the linguistic resource", re.IGNORECASE), 90),
    (
        re.compile(
            r"substantial portion of .{0,40}database|database contents|"
            r"individual contents\) is Adapted Material",
            re.IGNORECASE,
        ),
        95,
    ),
    (re.compile(r"work that uses", re.IGNORECASE), 50),
    (re.compile(r"combined work", re.IGNORECASE), 30),
    (re.compile(r"^\s*(?:\d+\.|[a-z]\))", re.IGNORECASE), 15),
)

_MAX_QUOTE = 400
# 短于该阈值的许可，问卷改吃全文，不再按 topic 配额摘片段。
SHORT_FULLTEXT_CHARS = 2500
SHORT_FULLTEXT_UNITS = 8
_FULLTEXT_EVIDENCE_ID = "e_full"
_VERSION_RE = re.compile(r"\bVersion\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)
_HEADING_SKIP = re.compile(
    r"^(preamble|license|licence|introduction|contents?|table of contents)\b",
    re.IGNORECASE,
)


def split_units(text: str) -> list[tuple[int, int, str]]:
    """按段落切分，过长再按句切。返回 (start, end, unit)。"""
    if not text:
        return []
    units: list[tuple[int, int, str]] = []
    parts = re.split(r"\n\s*\n", text)
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        end = idx + len(part)
        chunk = part.strip()
        if not chunk:
            cursor = end
            continue
        if len(chunk) <= 1200:
            units.append((idx, end, chunk))
        else:
            for m in re.finditer(r".+?(?:[.!?;]\s+|\n|$)", chunk, re.DOTALL):
                sub = m.group(0).strip()
                if not sub:
                    continue
                s = idx + m.start()
                e = idx + m.end()
                units.append((s, e, sub))
        cursor = end
    if not units and text.strip():
        units.append((0, len(text), text.strip()))
    return units


def is_short_fulltext(text: str) -> bool:
    """全文足够短时，问卷改读整篇。不按 short_id 特判。"""
    body = text or ""
    if not body.strip():
        return False
    if SHORT_FULLTEXT_CHARS <= 0:
        return False
    return (
        len(body) <= SHORT_FULLTEXT_CHARS
        and len(split_units(body)) <= SHORT_FULLTEXT_UNITS
    )


def make_fulltext_evidence(text: str) -> EvidenceItem:
    """合成单条全文证据，供短许可问卷引用。"""
    body = text or ""
    return EvidenceItem(
        id=_FULLTEXT_EVIDENCE_ID,
        topic="identity",
        quote=body,
        start=0,
        end=len(body),
        cue="fulltext",
        source="fulltext",
    )


def _clip(quote: str) -> str:
    quote = re.sub(r"\s+", " ", quote).strip()
    if len(quote) <= _MAX_QUOTE:
        return quote
    return quote[: _MAX_QUOTE - 1] + "…"


def extract_evidence(text: str) -> list[EvidenceItem]:
    """对每个文本单元打 topic；同一 (topic, span) 去重。"""
    items: list[EvidenceItem] = []
    seen: set[tuple[str, int, int]] = set()
    n = 0
    for start, end, unit in split_units(text):
        for topic, pattern, cue in PATTERNS:
            if not pattern.search(unit):
                continue
            key = (topic, start, end)
            if key in seen:
                continue
            seen.add(key)
            n += 1
            items.append(
                EvidenceItem(
                    id=f"e{n}",
                    topic=topic,
                    quote=_clip(unit),
                    start=start,
                    end=end,
                    cue=cue,
                )
            )
    items.extend(_gap_fill(text, items, seen, n))
    return items


def _gap_fill(
    text: str,
    items: list[EvidenceItem],
    seen: set[tuple[str, int, int]],
    n: int,
) -> list[EvidenceItem]:
    """问卷需要的 topic 若 0 条，用同一词表再扫一遍全文。"""
    present = {e.topic for e in items}
    extra: list[EvidenceItem] = []
    missing = GAP_TOPICS - present
    if not missing or not text:
        return extra
    for topic, pattern, cue in PATTERNS:
        if topic not in missing:
            continue
        for m in pattern.finditer(text):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 160)
            key = (topic, start, end)
            if key in seen:
                continue
            seen.add(key)
            n += 1
            extra.append(
                EvidenceItem(
                    id=f"e{n}",
                    topic=topic,
                    quote=_clip(text[start:end]),
                    start=start,
                    end=end,
                    cue=cue,
                )
            )
            break
    return extra


def _quote_salience(item: EvidenceItem) -> int:
    """把区分性强的条款排到问卷前面，避免前言占满配额。"""
    quote = item.quote or ""
    q = quote.lower()
    score = 0
    if item.topic in COMBINE_TOPICS or item.topic == "exception_or_separate_work":
        for pattern, weight in _COMBINE_SALIENT:
            if pattern.search(quote):
                score += weight
    if item.topic in {"compat_convert", "copyleft_sharealike"}:
        if "compatible" in q or "affero" in q or "opt to apply" in q:
            score += 80
    if item.topic == "patent":
        if "patent license" in q:
            score += 80
        if "non-exclusive" in q or "royalty-free" in q:
            score += 40
    if item.topic == "output_generate":
        if "output from running" in q or "constitutes a covered work" in q:
            score += 100
        elif "output" in q:
            score += 30
    if item.topic == "commercial":
        if "non-commercial" in q or "not for commercial" in q:
            score += 80
    return score


_OPERATIVE_GRANT_RE = re.compile(
    r"hereby grants|permission is hereby granted|sublicensable|without restriction",
    re.IGNORECASE,
)


def supplement_relicense_evidence(
    evidence: list[EvidenceItem],
    base: list[EvidenceItem],
    *,
    limit: int = 6,
) -> list[EvidenceItem]:
    """把已抽到、但没进修改包的授予和附文补进换证引文。最多 limit 条。"""
    grouped: dict[tuple[Optional[int], Optional[int]], list[EvidenceItem]] = {}
    order: list[tuple[Optional[int], Optional[int]]] = []
    for item in evidence:
        span = (item.start, item.end)
        if span not in grouped:
            order.append(span)
        grouped.setdefault(span, []).append(item)
    seen = {(item.start, item.end) for item in base}
    ranked: list[tuple[int, int, EvidenceItem]] = []
    for index, span in enumerate(order):
        if span in seen:
            continue
        items = grouped[span]
        topics = {item.topic for item in items}
        host = next(
            (item for item in items if item.topic in {"grant", "notice_attribution"}),
            None,
        )
        if host is None:
            continue
        quote = host.quote or ""
        if re.search(r"\bmeans\b", quote, re.IGNORECASE) and not _OPERATIVE_GRANT_RE.search(quote):
            continue
        window = order[index : index + 4]
        window_topics = {item.topic for nxt in window for item in grouped[nxt]}
        operative = bool(_OPERATIVE_GRANT_RE.search(quote))
        attached = "notice_attribution" in topics and (
            "redistribute" in window_topics or "modify_adapt" in window_topics
        )
        same_unit = bool({"redistribute", "modify_adapt"} & topics)
        if not (operative or attached or same_unit):
            continue
        priority = 0 if operative else 1 if attached else 2
        ranked.append((priority, index, host))
    ranked.sort()
    return [item for _priority, _index, item in ranked[:limit]]


def evidence_for_topics(
    evidence: list[EvidenceItem],
    topics,
    limit: int = 15,
    per_topic: int = 2,
) -> list[EvidenceItem]:
    """按主题配额取引文，避免前文 grant 挤掉 notice/copyleft/aggregation。"""
    wanted = list(topics)
    buckets: dict[str, list[EvidenceItem]] = {t: [] for t in wanted}
    for e in evidence:
        if e.topic in buckets:
            buckets[e.topic].append(e)
    for topic, items in buckets.items():
        items.sort(key=lambda e: (-_quote_salience(e), e.start if e.start is not None else 0))
    picked: list[EvidenceItem] = []
    seen: set[str] = set()

    def _take(topic: str, cap: int) -> None:
        have = sum(1 for x in picked if x.topic == topic)
        for e in buckets.get(topic) or []:
            if have >= cap or len(picked) >= limit:
                return
            if e.id in seen:
                continue
            picked.append(e)
            seen.add(e.id)
            have += 1

    for topic in wanted:
        _take(topic, 1)
    if per_topic > 1:
        for topic in wanted:
            _take(topic, per_topic)
    ranked_rest = sorted(
        (e for e in evidence if e.topic in buckets and e.id not in seen),
        key=lambda e: (-_quote_salience(e), e.start if e.start is not None else 0),
    )
    for e in ranked_rest:
        if len(picked) >= limit:
            break
        picked.append(e)
        seen.add(e.id)
    return picked


def has_combine_evidence(evidence: list[EvidenceItem]) -> bool:
    return any(e.topic in COMBINE_TOPICS for e in evidence)


_PREFIX_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("perpetual", re.compile(r"\bperpetual\b", re.IGNORECASE)),
    ("worldwide", re.compile(r"\bworldwide\b", re.IGNORECASE)),
    ("non-exclusive", re.compile(r"\bnon-?exclusive\b", re.IGNORECASE)),
    ("no-charge", re.compile(r"\bno[ -]?charge\b|free of charge", re.IGNORECASE)),
    ("royalty-free", re.compile(r"\broyalty[ -]?free\b", re.IGNORECASE)),
    ("irrevocable", re.compile(r"\birrevocable\b", re.IGNORECASE)),
    ("non-transferable", re.compile(r"\bnon-?transferable\b", re.IGNORECASE)),
    ("non-sublicensable", re.compile(r"\bnon-?sublicens(?:able|ing)\b", re.IGNORECASE)),
    ("revocable", re.compile(r"\brevocable\b", re.IGNORECASE)),
    ("sublicensable", re.compile(r"\bsublicensable\b", re.IGNORECASE)),
    ("any-person", re.compile(r"\bany person\b", re.IGNORECASE)),
]
# 只有本许可自己授予专利才算。不把「他人取得的 patent license」算成授予。
_PATENT_GRANT_RE = re.compile(
    r"grants?\s+(?:to\s+)?you\b.{0,180}?\bpatent\b|"
    r"hereby grants\b.{0,200}?\bpatent licen[cs]es?\b|"
    r"grant of patent licen[cs]e\b|"
    r"licensor grants you\b.{0,180}?\bpatent\b",
    re.IGNORECASE | re.DOTALL,
)
_PATENT_DENIED_RE = re.compile(
    r"no patent (?:rights|licen[cs]e)|"
    r"does not grant.{0,40}patent|"
    r"without any.{0,20}patent|"
    r"patent(?: and trademark)? rights are not licen[cs]ed|"
    r"\bno\b.{0,80}patent rights are granted",
    re.IGNORECASE | re.DOTALL,
)
# 限定范围的排除（other than / except licensed claims）≠ 全文不授予专利。
_PATENT_DENY_SCOPED_RE = re.compile(
    r"\bother than\b|"
    r"\bexcept\b.{0,60}(?:licensed claims|section\s+\d)|"
    r"licensed claims defined",
    re.IGNORECASE | re.DOTALL,
)
# 商标权本身被保留。使用权禁止、通知里的商标声明不算。
_TRADEMARK_RIGHT_RESERVED_RE = re.compile(
    r"trademark rights are reserved|"
    r"reserves?\b.{0,40}trademark rights|"
    r"trademark rights? (?:are|is) not (?:assigned|transferred|waived)",
    re.IGNORECASE | re.DOTALL,
)
_NC_ALLOW_RE = re.compile(
    r"commercial or non-?commercial|commercial and non-?commercial",
    re.IGNORECASE,
)
# 某一种履行方式仅限非商业，不是收回作品的商业使用。
_NC_ALTERNATIVE_RE = re.compile(
    r"\balternative\b.{0,120}\bonly for non-?commercial\b",
    re.IGNORECASE,
)
# 作品级版权放弃 / 公有领域捐献（所有权原子），不是「copyright license」授权句。
_COPYRIGHT_DEDICATED_POS_RE = re.compile(
    r"released into the public domain|"
    r"place[sd]?\b.{0,60}(?:in(?:to)?|into) the public domain|"
    r"dedicate[sd]?\b.{0,120}(?:copyright|work|material|software|database|content)"
    r".{0,80}public domain|"
    r"dedicate[sd]?\b.{0,60}(?:any and all )?copyright interest|"
    r"dedication (?:of .{0,60} )?(?:to the )?public domain|"
    r"waive[sd]?,?\s*(?:abandons?,?\s*(?:and\s+)?)?(?:surrenders?\s+)?"
    r".{0,100}copyright|"
    r"waive[sd]?\b.{0,80}(?:all .{0,30})?copyright|"
    r"\bno rights reserved\b",
    re.IGNORECASE | re.DOTALL,
)
_COPYRIGHT_DEDICATED_NEG_RE = re.compile(
    r"text of .{0,120}licen[cs]e|"
    r"(?:public\s+)?licen[cs]es?\s+is dedicated|"
    r"does not form part of .{0,80}(?:public\s+)?licen|"
    r"waive[sd]?\b.{0,120}(?:technical modifications|"
    r"effective technological measures|moral rights|publicity|privacy)|"
    r"copyright notice|"
    r"(?:grant|grants|granted|granting)\b.{0,60}copyright licen|"
    r"copyright licen[cs]e to|"
    r"permissions under copyright|"
    r"licen[cs]e under copyright",
    re.IGNORECASE | re.DOTALL,
)


def _copyright_dedicated(text: str) -> bool:
    """本作品版权是否被放弃/捐公有领域。许可文本脚注、局部豁免不算。"""
    body = text or ""
    for m in _COPYRIGHT_DEDICATED_POS_RE.finditer(body):
        start = max(0, m.start() - 100)
        end = min(len(body), m.end() + 100)
        if _COPYRIGHT_DEDICATED_NEG_RE.search(body[start:end]):
            continue
        return True
    return False


_SOURCE_DELIVERY_RE = re.compile(
    r"corresponding source|network server at no charge|"
    r"access to copy.{0,80}at no charge",
    re.IGNORECASE,
)
_PATENT_UNIT_RE = re.compile(
    r"\bpatent license\b|\bpatent claims\b|grants? you a.{0,60}patent",
    re.IGNORECASE,
)
_GRANT_UNIT_RE = re.compile(
    r"permission is hereby granted|all rights granted|"
    r"copyright license|"
    r"hereby grants?(?:\s+to)?(?:\s+you)?\s+a (?!no rights)"
    r"(?:perpetual|worldwide|non-exclusive|royalty-free|revocable|"
    r"irrevocable|non-sublicensable|non-transferable|limited)|"
    r"hereby grants?(?:\s+to)?(?:\s+you)?\s+a.{0,80}(?:copyright|patent) license|"
    # 无 hereby：Licensor/Owner grants you a … license（研究型模型协议常见）
    r"\bgrants?\s+(?:to\s+)?you\s+a\b(?!\s+no\s+rights).{0,200}\blicen[cs]e\b|"
    # 被动：You are granted a … license
    r"\byou are granted\s+a\b(?!\s+no\s+rights).{0,200}\blicen[cs]e\b",
    re.IGNORECASE | re.DOTALL,
)
_NO_RIGHTS_GRANT_RE = re.compile(r"grants? you no rights", re.IGNORECASE)
# 版权授予窗内的权能（排除 AUP / Attachment）。
_AUP_SEGMENT_RE = re.compile(
    r"use-based restrictions|attachment\s+[a-z]\b|prohibited uses|"
    r"you agree not to use",
    re.IGNORECASE,
)
_GRANT_USE_RE = re.compile(
    r"\breproduce\b|"
    r"copyright licen[cs]e to\b.{0,160}\b(?:use|reproduce)\b|"
    r"permission is hereby granted\b.{0,400}\b(?:to use\b|rights to use\b)|"
    r"\blicen[cs]e to use\b|"
    r"\brights to use\b|"
    r"reproduce and Share\b",
    re.IGNORECASE | re.DOTALL,
)
_GRANT_MODIFY_RE = re.compile(
    r"\bprepare\b|"
    r"copyright licen[cs]e to\b.{0,200}\bprepare\b|"
    r"permission is hereby granted\b.{0,400}\bmodify\b|"
    r"\bto modify\b|"
    r"prepare Derivative|"
    r"create derivative works|"
    r"make modifications\b|"
    r"use, copy, modify|"
    r"produce,?\s*reproduce,?\s*and Share Adapted|"
    r"Share Adapted Material|"
    r"produce.{0,40}Adapted Material",
    re.IGNORECASE | re.DOTALL,
)
_GRANT_REDISTRIBUTE_RE = re.compile(
    r"\bdistribute\b|\bDistribution\b|"
    r"copyright licen[cs]e to\b.{0,200}\bdistribute\b|"
    r"permission is hereby granted\b.{0,400}\b(?:publish|distribute)\b|"
    r"\band Share\b|\bto Share\b",
    re.IGNORECASE | re.DOTALL,
)
_GRANT_SUBLICENSE_RE = re.compile(
    # 正向授予列举中的 sublicense；排除 non-sublicensable / may not sublicense
    r"(?:to |and |, )sublicense\b|"
    r"\brights to use,.{0,80}\bsublicense\b|"
    r"permission is hereby granted\b.{0,400}\bsublicense\b",
    re.IGNORECASE | re.DOTALL,
)
_GRANT_MERGE_RE = re.compile(
    r"\bto merge\b|"
    r"modify, merge|"
    r"permission is hereby granted\b.{0,400}\bmerge\b|"
    r"\brights to use, copy, modify, merge\b|"
    # 版权授予句制备/分发衍生作品 ⇒ 合入他作权能（OpenRAIL/Apache 等无 MIT 字面 merge）
    r"copyright licen[cs]e to\b.{0,240}\bprepare\b|"
    r"prepare Derivative Works\b|"
    r"reproduce,?\s*prepare,.{0,160}Derivatives?\b|"
    r"prepare,?\s*publicly display,?\s*publicly perform.{0,120}distribute",
    re.IGNORECASE | re.DOTALL,
)
# 无义务骨架的极宽松套话（非公有领域捐献专名）。
_ULTRA_PERMISSIVE_RE = re.compile(
    r"do what the (?:fuck|f\*+ck) you want|"
    r"you just do what .{0,40} you want|"
    r"do whatever you want to",
    re.IGNORECASE,
)
_SUBLICENSE_BAN_RE = re.compile(
    r"sublicens(?:ing|es)? is not allowed|"
    r"\bsublicensing is not allowed\b|"
    r"no (?:right to )?sublicens|"
    r"may not sublicense|"
    r"may not assign or sublicense|"
    r"you may not sublicense|"
    r"you may not assign or sublicense|"
    r"nothing in this licen[cs]e permits.{0,100}sublicens|"
    r"does not (?:permit|allow).{0,40}sublicens|"
    r"does not.{0,80}grant.{0,80}(?:any )?(?:rights? to )?sublicens|"
    r"\bnon-?sublicens(?:able|ing)\b",
    re.IGNORECASE | re.DOTALL,
)
_SUBLICENSE_GRANT_RE = re.compile(
    r"(?:to |and/or |right to )sublicense|"
    r"including without limitation the rights to.{0,80}sublicense|"
    r"copyright license to.{0,200}sublicense",
    re.IGNORECASE | re.DOTALL,
)
_MARK_CHANGES_RE = re.compile(
    r"stating that you (?:modified|changed)|"
    r"modified files to carry prominent notices|"
    r"prominent notices stating that you|"
    r"indicate if you modified|"
    r"retain an indication of any previous modifications|"
    # 须告知已修改（不依赖某一专名标题）
    r"inform(?:s|ing)?\b.{0,100}(?:recipients|that you).{0,60}modified|"
    r"(?:notice|notices)\b.{0,100}(?:that you|you have)\s+modified|"
    r"reasonably calculated to inform.{0,100}modified|"
    r"carry a prominent\b.{0,80}notice.{0,100}modified|"
    r"document how it differs|"
    r"describ(?:e|es|ing) how\b.{0,40}(?:differs|changes)",
    re.IGNORECASE | re.DOTALL,
)
# license 与 agreement 同等。动词只限 provide / give / include / attach / make available。
# 「另签协议」「终止后另签」不含 copy of this，对不上。
_ATTACH_DOC = r"(?:licen[cs]e|agreement)"
_ATTACH_COPY = rf"(?:a |the )?(?:complete )?copy of this {_ATTACH_DOC}"
_ATTACH_LICENSE_RE = re.compile(
    rf"(?:give|provide|include|attach|make available)\b.{{0,160}}{_ATTACH_COPY}|"
    rf"give.{{0,80}}recipients.{{0,80}}{_ATTACH_COPY}|"
    r"a copy of this public license|"
    rf"{_ATTACH_COPY}.{{0,40}}(?:must |shall )?be (?:included|provided|attached|given)|"
    rf"include a complete copy of this {_ATTACH_DOC}|"
    rf"including a complete copy of this {_ATTACH_DOC}|"
    r"indicate.{0,80}licensed under this(?:\s+public)?\s+licen[cs]e|"
    r"include the text of,?\s+or the (?:uri|url|hyperlink)\s+to,?\s+this(?:\s+public)?\s+licen[cs]e|"
    # BSD/OpenSSL 条件簇：保留 notice 与 this list of conditions
    r"this list of conditions|"
    r"reproduce the above copyright notice, this list of conditions|"
    r"retain the above copyright notice, this list of conditions",
    re.IGNORECASE | re.DOTALL,
)
# 「仅仅附上协议并不足以适用另一许可」本身不是附证义务。
_NOT_SUFFICIENT_ATTACH_RE = re.compile(
    r"not sufficient to licen",
    re.IGNORECASE,
)
# 源码形态须附完整许可文本（与仅「binary 须 reproduce notice」区分）
_SOURCE_ATTACH_LICENSE_RE = re.compile(
    r"(?:in )?source code form.{0,220}"
    r"(?:(?:complete )?copy of this licen[cs]e|including a complete copy)",
    re.IGNORECASE | re.DOTALL,
)
_BINARY_FORM_CLAUSE_RE = re.compile(
    r"(?:compiled or )?object code form|"
    r"\bin binary form\b|"
    r"Redistributions in binary form",
    re.IGNORECASE,
)
_BINARY_ATTACH_LICENSE_RE = re.compile(
    r"(?:compiled or )?object code form.{0,220}(?:complete )?copy of this licen|"
    r"\bin binary form\b.{0,220}(?:complete )?copy of this licen|"
    r"binary form must.{0,120}(?:include|attach|provide).{0,80}(?:copy of )?(?:this )?licen",
    re.IGNORECASE | re.DOTALL,
)
_RETAIN_NOTICE_RE = re.compile(
    r"(?:retain|keep|preserve)\b.{0,100}(?:copyright|attribution|patent|trademark).{0,40}notices?|"
    r"(?:copyright|attribution) notice.{0,60}(?:shall be )?(?:included|retained)|"
    r"the above copyright notice|"
    # Attribution-style: retain a supplied list that includes copyright / license notices
    r"(?:retain|keep|preserve)\b.{0,400}copyright notice|"
    r"(?:retain|keep|preserve)\b.{0,400}notice that refers to this(?:\s+public)?\s+licen[cs]e|"
    r"a notice that refers to this(?:\s+public)?\s+licen[cs]e|"
    # 不得删除已有声明。不要求 retain 这个词。
    r"(?:may |shall |must )?not (?:remove|alter)\b.{0,80}(?:copyright|patent|trademark|attribution).{0,30}notices?|"
    r"not (?:remove|alter) any\b.{0,40}notices",
    re.IGNORECASE | re.DOTALL,
)
# 说明作品已被修改。不得改许可文本 / 不得删声明不算。
_DESCRIBE_CHANGES_RE = re.compile(
    r"document how it differs|"
    r"describ(?:e|es|ing) how\b.{0,40}(?:differs|changes)|"
    r"notices? stating (?:the |that you )(?:changes|changed|modified)",
    re.IGNORECASE | re.DOTALL,
)

def _unit_kind(unit: str) -> str:
    if _NO_RIGHTS_GRANT_RE.search(unit):
        return "other"
    if _SOURCE_DELIVERY_RE.search(unit):
        return "source"
    if _PATENT_UNIT_RE.search(unit) and not re.search(r"copyright license", unit, re.I):
        return "patent"
    if _GRANT_UNIT_RE.search(unit):
        return "grant"
    return "other"


_SECTION_PIN_RE = re.compile(r"(\d+\.[a-z])\.", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")
_GRANT_DIFFERENT_TERMS_RE = re.compile(
    r"may provide additional or different licen[cs]e terms",
    re.IGNORECASE,
)
_MOD_OR_DERIVATIVE_RE = re.compile(
    r"modif(?:y|ied|ications)|derivative",
    re.IGNORECASE,
)
_RELICENSE_BLOCK_RE = re.compile(
    r"such licen[cs]e terms (?:limit|require)|"
    r"respecting paragraph|"
    r"use-based restrictions|"
    r"except as set forth",
    re.IGNORECASE,
)
_AMALGAMATE_FOLLOWS_MODIFY_RE = re.compile(
    r"modifications?, or for any such "
    r"(?:derivative works|derivatives of the model|model derivatives)"
    r" as a whole",
    re.IGNORECASE,
)


def iter_license_sentences(text: str) -> list[str]:
    """按句切开。小节号如 4.a. 不当成句号，避免 OpenRAIL 套话被截断。"""
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return []
    protected = _SECTION_PIN_RE.sub(r"\1∯", flat)
    parts = _SENTENCE_SPLIT_RE.split(protected)
    return [part.replace("∯", ".").strip() for part in parts if part.strip()]


_DEF_MARK_RE = re.compile(
    r"\bmeans\b|\bdefined as\b|\brefers to\b|\bshall mean\b|"
    r"\bconsist(?:s|ing)? of\b|\bcompris(?:e|es|ing)\b",
    re.IGNORECASE,
)
_NOT_THE_LICENSED_OBJECT_RE = re.compile(
    r"\b(?:is|are) not licensed\b|\bnot covered by this\b|\bdo(?:es)? not include\b",
    re.IGNORECASE,
)
_OBLIGATION_INCLUDE_RE = re.compile(
    r"\b(?:must|shall|agree to)\b",
    re.IGNORECASE,
)
_SOFTWARE_BODY_RE = re.compile(r"\bsource code\b|\bscripts?\b", re.IGNORECASE)
_SOFTWARE_WORD_RE = re.compile(r"\bsoftware\b", re.IGNORECASE)
_OF_THE_SOFTWARE_RE = re.compile(r"\bof (?:the|this) software\b", re.IGNORECASE)
_NC_WORD_RE = re.compile(r"\bnon-?commercial\b", re.IGNORECASE)
_NC_LIMIT_RE = re.compile(
    r"\bonly\b|\bsolely\b|\blimited\b|\brestrict|\bmeans\b|\bpurpose|"
    r"\bshall not\b|\bmay not\b|\bprohibited\b|\bforbidden\b|\bnot for commercial\b",
    re.IGNORECASE,
)
_NC_NEG_RE = re.compile(
    r"\bnot\b|\bno\b|\bprohibited\b|\bforbidden\b|\bbars?\b",
    re.IGNORECASE,
)
_SOLE_DIRECT_RE = re.compile(r"directly granted by", re.IGNORECASE)
_SOLE_GRANT_WORD_RE = re.compile(r"\blicen[cs]e\b|\bgrant\b", re.IGNORECASE)
_SOLE_OBTAIN_RE = re.compile(r"\bmust\b|\bshall\b", re.IGNORECASE)
_SOLE_OBTAIN_VERB_RE = re.compile(r"\brequest\b|\bobtain\b|\bseek\b", re.IGNORECASE)
_SOLE_FROM_RE = re.compile(r"\blicensor\b|\bgrantor\b|granted by", re.IGNORECASE)
_LICENSE_TRAVEL_RE = re.compile(
    r"copy of this (?:licen[cs]e|agreement)",
    re.IGNORECASE,
)
_ND_WITHDRAW_RE = re.compile(
    r"do not have permission|you may not|you shall not|no permission|"
    r"not permitted|but not share|without the right to (?:share|distribut|publish)|"
    r"\bprohibited\b|\bforbidden\b|\bno right to publish\b",
    re.IGNORECASE,
)
_ND_OBJECT_RE = re.compile(
    r"adapted material|derivative|modified material|modifications|extracted model",
    re.IGNORECASE,
)
_ND_SHARE_RE = re.compile(
    r"\bshare\b|\bdistribut|\bpublish(?:ing|ed|es)?\b|\bpublication\b",
    re.IGNORECASE,
)
_ND_CONDITIONAL_RE = re.compile(
    r"\bif you\b|provided that|as long as|so long as",
    re.IGNORECASE,
)
_ND_ANCHOR_NEG_RE = re.compile(
    r"\bnot\b|\bno\b|\bwithout\b|\bdo not\b|\bmay not\b|\bshall not\b|"
    r"\bprohibited\b|\bforbidden\b",
    re.IGNORECASE,
)
_MODEL_OBJECT_RE = re.compile(
    r"\bweights?\b|\bcheckpoints?\b|\bparameters?\b",
    re.IGNORECASE,
)
_DATA_OBJECT_RE = re.compile(
    r"\bdatabase\b|\bdatasets?\b|\bcorpus\b|\bliterary\b|\bartistic work\b",
    re.IGNORECASE,
)
_NC_BAR_RE = re.compile(
    r"\bforbidden\b|\bprohibited\b|\bbarred\b|\bnot permitted\b",
    re.IGNORECASE,
)
_TM_WORD_RE = re.compile(
    r"\btrademarks?\b|\btrade ?names?\b|\btrade ?marks?\b",
    re.IGNORECASE,
)
_TM_USE_BAN_RE = re.compile(
    r"no trademark licen|"
    r"not (?:been )?granted|"
    r"does not grant|"
    r"nothing in this|"
    r"no licen[cs]e is granted|"
    r"shall not|"
    r"may not|"
    r"not be used|"
    r"used to endorse|"
    r"publicity|advertising|marketing|endorse|promote|"
    r"reserved font name|reserved name",
    re.IGNORECASE,
)
_TM_GPL_MENU_RE = re.compile(
    r"declining to grant rights under trademark|you may supplement the terms",
    re.IGNORECASE,
)
_TM_NOTICE_RE = re.compile(r"\bretain\b|\bnotice", re.IGNORECASE)


def _is_defining(sentence: str) -> bool:
    """定义或组成句。义务性的 include、以及声明某物不在本许可内的句子不算。"""
    if _NOT_THE_LICENSED_OBJECT_RE.search(sentence):
        return False
    if _DEF_MARK_RE.search(sentence):
        return True
    if _OBLIGATION_INCLUDE_RE.search(sentence):
        return False
    return re.search(r"\bincludes?\b", sentence, re.IGNORECASE) is not None


def _software_defined_in_sentence(sentence: str) -> bool:
    """定义句把源代码、软件或脚本写成许可对象。"""
    if not _is_defining(sentence):
        return False
    if _SOFTWARE_BODY_RE.search(sentence):
        return True
    if not _SOFTWARE_WORD_RE.search(sentence):
        return False
    stripped = _OF_THE_SOFTWARE_RE.sub(" ", sentence)
    return _SOFTWARE_WORD_RE.search(stripped) is not None


def _model_defined_in_sentence(sentence: str) -> bool:
    return bool(_is_defining(sentence) and _MODEL_OBJECT_RE.search(sentence))


def _data_defined_in_sentence(sentence: str) -> bool:
    return bool(_is_defining(sentence) and _DATA_OBJECT_RE.search(sentence))


def _software_anchor_in_sentence(sentence: str) -> bool:
    """向量复核：句中出现源代码、脚本，或把 software 写成被定义或包含的对象。"""
    if _SOFTWARE_BODY_RE.search(sentence):
        return True
    return bool(
        _SOFTWARE_WORD_RE.search(sentence)
        and re.search(r"\bmeans\b|\binclude[sd]?\b|\bconsist|\bcompris", sentence, re.I)
    )


def _non_commercial_in_sentence(sentence: str) -> bool:
    if _NC_ALLOW_RE.search(sentence) or _NC_ALTERNATIVE_RE.search(sentence):
        return False
    if _NC_WORD_RE.search(sentence) and _NC_LIMIT_RE.search(sentence):
        return True
    if re.search(r"\bcommercial\b", sentence, re.IGNORECASE) and _NC_BAR_RE.search(
        sentence
    ):
        return True
    return bool(re.search(r"\bnot for commercial\b", sentence, re.IGNORECASE))


def _non_commercial_anchor_in_sentence(sentence: str) -> bool:
    if _NC_ALLOW_RE.search(sentence) or _NC_ALTERNATIVE_RE.search(sentence):
        return False
    if _NC_WORD_RE.search(sentence) or re.search(
        r"\bnot for commercial\b", sentence, re.IGNORECASE
    ):
        return True
    return bool(
        re.search(r"\bcommercial\b", sentence, re.IGNORECASE)
        and _NC_NEG_RE.search(sentence)
    )


def text_has_non_commercial(text: str) -> bool:
    """某一句把使用限制在非商业。并列允许商业与非商业的那一句本身不算。"""
    return any(_non_commercial_in_sentence(s) for s in iter_license_sentences(text or ""))


_COMMERCIAL_SCALE_RE = re.compile(r"\bmonthly active users\b", re.IGNORECASE)
_COMMERCIAL_THRESHOLD_RE = re.compile(
    r"\bgreater than\b|\bmore than\b|\bexceeds?\b|\bin excess of\b",
    re.IGNORECASE,
)
_COMMERCIAL_GATE_RE = re.compile(
    r"\bmust request a licen[cs]e\b|"
    r"\bnot authorized to exercise\b|"
    r"\bsole discretion\b",
    re.IGNORECASE,
)
_TERRITORY_DEF_RE = re.compile(
    r"\bterritory\b.{0,80}\b(?:shall mean|means)\b|"
    r"\b(?:shall mean|means)\b.{0,40}\bterritory\b",
    re.IGNORECASE,
)
_TERRITORY_EXCLUDE_RE = re.compile(r"\bexclud(?:e|ing|es)\b", re.IGNORECASE)


def _commercial_scale_limited(sentence: str) -> bool:
    return bool(
        _COMMERCIAL_SCALE_RE.search(sentence)
        and _COMMERCIAL_THRESHOLD_RE.search(sentence)
        and _COMMERCIAL_GATE_RE.search(sentence)
    )


def _commercial_territory_limited(sentence: str) -> bool:
    return bool(
        _TERRITORY_DEF_RE.search(sentence)
        and _TERRITORY_EXCLUDE_RE.search(sentence)
        and re.search(r"\bworldwide\b|\bcountr|\bregion\b", sentence, re.IGNORECASE)
    )


def _commercial_use_limited_in_sentence(sentence: str) -> bool:
    """商业使用权本身的规模门槛或地域排除。禁止用途和使用政策不算。"""
    return _commercial_scale_limited(sentence) or _commercial_territory_limited(sentence)


def _commercial_use_limited_anchor_in_sentence(sentence: str) -> bool:
    if _commercial_use_limited_in_sentence(sentence):
        return True
    if re.search(r"\bjurisdiction\b", sentence, re.IGNORECASE):
        return False
    if _COMMERCIAL_SCALE_RE.search(sentence) and (
        _COMMERCIAL_THRESHOLD_RE.search(sentence) or _COMMERCIAL_GATE_RE.search(sentence)
    ):
        return True
    return bool(
        re.search(r"\bterritory\b", sentence, re.IGNORECASE)
        and _TERRITORY_EXCLUDE_RE.search(sentence)
    )


def _sole_separate_grant(sentence: str) -> bool:
    low = sentence.lower()
    grant_word = _SOLE_GRANT_WORD_RE.search(sentence) is not None
    named_separate = grant_word and (
        "separate" in low or "additional" in low or "further" in low
    )
    must_obtain = bool(
        grant_word
        and _SOLE_OBTAIN_RE.search(sentence)
        and _SOLE_OBTAIN_VERB_RE.search(sentence)
    )
    only_from = "granted only by" in low or "only be granted by" in low
    return named_separate or must_obtain or only_from


def _rights_remain_upstream(sentence: str) -> bool:
    """接收方不能凭收到文本取得权利，且这些权利仍留在许可人处。

    「许可人声明对产出不享有权利」和「不接受条款则不授予权利」不是这条。
    """
    low = sentence.lower()
    if "claims no rights" in low or "do not agree" in low:
        return False
    no_travel = any(
        phrase in low
        for phrase in (
            "do not receive",
            "do not acquire",
            "acquire no rights",
            "do not take rights",
            "no rights are conveyed",
            "no rights pass",
            "rights do not travel",
        )
    )
    stays = any(
        phrase in low
        for phrase in ("remain with", "retained by", "remains with", "stay with")
    )
    return no_travel and stays


def _sole_source_in_sentence(sentence: str) -> bool:
    if _SOLE_DIRECT_RE.search(sentence):
        return True
    if _LICENSE_TRAVEL_RE.search(sentence) and not _sole_separate_grant(sentence):
        return False
    if _sole_separate_grant(sentence) and _SOLE_FROM_RE.search(sentence):
        return True
    return _rights_remain_upstream(sentence)


def _model_anchor_in_sentence(sentence: str) -> bool:
    if _NOT_THE_LICENSED_OBJECT_RE.search(sentence):
        return False
    return _MODEL_OBJECT_RE.search(sentence) is not None


def _data_anchor_in_sentence(sentence: str) -> bool:
    if _NOT_THE_LICENSED_OBJECT_RE.search(sentence):
        return False
    return _DATA_OBJECT_RE.search(sentence) is not None


def _sole_source_anchor_in_sentence(sentence: str) -> bool:
    if _rights_remain_upstream(sentence) or _sole_source_in_sentence(sentence):
        return True
    if _LICENSE_TRAVEL_RE.search(sentence) and not (
        _SOLE_DIRECT_RE.search(sentence) or _sole_separate_grant(sentence)
    ):
        return False
    low = sentence.lower()
    party = re.search(r"recipient|third[- ]party|downstream", low) is not None
    return bool(_SOLE_FROM_RE.search(sentence) and party)


def _nd_in_sentence(sentence: str) -> bool:
    if _ND_CONDITIONAL_RE.search(sentence) and not _ND_WITHDRAW_RE.search(sentence):
        return False
    return bool(
        _ND_WITHDRAW_RE.search(sentence)
        and _ND_OBJECT_RE.search(sentence)
        and _ND_SHARE_RE.search(sentence)
    )


def _nd_anchor_in_sentence(sentence: str) -> bool:
    if _ND_CONDITIONAL_RE.search(sentence) and not re.search(
        r"do not have permission|but not share", sentence, re.IGNORECASE
    ):
        return False
    return bool(
        _ND_ANCHOR_NEG_RE.search(sentence)
        and _ND_OBJECT_RE.search(sentence)
        and _ND_SHARE_RE.search(sentence)
    )


def _endorsement_ban_in_sentence(sentence: str) -> bool:
    """禁止用名称或商标做背书、宣传。不一定出现 trademark 这个词。"""
    low = sentence.lower()
    if not re.search(r"endors|promot|publicity|advertis", low):
        return False
    if "neither the name" in low or "the names" in low:
        return "used to" in low or "be used" in low
    return bool(
        re.search(
            r"not be used|must not|shall not|may not|does not grant|"
            r"not permitted|prohibited|forbidden",
            low,
        )
    )


def _trademark_use_in_sentence(sentence: str) -> bool:
    if _TM_GPL_MENU_RE.search(sentence):
        return False
    if re.search(r"\bReserved Font Name\b|\bReserved Name\b", sentence, re.IGNORECASE):
        return True
    if _endorsement_ban_in_sentence(sentence):
        return True
    if not _TM_WORD_RE.search(sentence):
        return False
    if _TM_NOTICE_RE.search(sentence) and not re.search(
        r"publicity|endors|promot|advertis|no trademark|does not grant|"
        r"not granted|not licensed|shall not|may not|no licen",
        sentence,
        re.IGNORECASE,
    ):
        return False
    low = sentence.lower()
    if _TM_USE_BAN_RE.search(sentence):
        return True
    if "not licensed" in low or "not licenced" in low:
        return True
    withheld = any(word in low for word in ("waived", "abandoned", "surrendered"))
    return bool(("no trademark" in low or "no trade mark" in low) and withheld)


def _trademark_use_anchor_in_sentence(sentence: str) -> bool:
    return _trademark_use_in_sentence(sentence)


def _defined_term(sentence: str) -> str | None:
    quoted = re.search(r'"([^"]{1,60})"\s+means\b', sentence, re.IGNORECASE)
    if quoted:
        return quoted.group(1)
    titled = re.search(r"\b([A-Z][\w-]{1,40})\s+means\b", sentence)
    return titled.group(1) if titled else None


def _next_sentence_excludes_defined_object(sentences: list[str], index: int) -> bool:
    """定义句的下一句若声明该定义项不在本许可内，则这一定义不作数。"""
    if index + 1 >= len(sentences):
        return False
    following = sentences[index + 1]
    if not _NOT_THE_LICENSED_OBJECT_RE.search(following):
        return False
    term = _defined_term(sentences[index])
    if not term:
        return False
    return re.search(rf"\b{re.escape(term)}\b", following, re.IGNORECASE) is not None


def _any_sentence(text: str, predicate) -> bool:
    return any(predicate(sentence) for sentence in iter_license_sentences(text or ""))


def _any_kept_definition(text: str, predicate) -> bool:
    sentences = iter_license_sentences(text or "")
    return any(
        predicate(sentence) and not _next_sentence_excludes_defined_object(sentences, index)
        for index, sentence in enumerate(sentences)
    )


def different_terms_flags(text: str) -> tuple[bool, bool]:
    """返回 (可无条件换许可, amalgamate 应继承 modify)。

    yes 只认「may provide additional or different license terms」，且同句对象是
    修改版或衍生作品。同句若限制新许可本身，或写了除外条款，则不投 yes。
    amalgamate 继承只认修改版与 as a whole 写在同一句的套话，不受上述否决影响。
    """
    clean_yes = False
    follows = False
    for sentence in iter_license_sentences(text):
        if not _GRANT_DIFFERENT_TERMS_RE.search(sentence):
            continue
        blocked = _RELICENSE_BLOCK_RE.search(sentence) is not None
        if not blocked and _MOD_OR_DERIVATIVE_RE.search(sentence):
            clean_yes = True
        if _AMALGAMATE_FOLLOWS_MODIFY_RE.search(sentence):
            follows = True
    return clean_yes, follows


def amalgamate_follow_sentences(text: str) -> list[str]:
    """修改版与衍生整体写在同一句的完整句子。"""
    found: list[str] = []
    seen: set[str] = set()
    for sentence in iter_license_sentences(text or ""):
        if not _AMALGAMATE_FOLLOWS_MODIFY_RE.search(sentence):
            continue
        clean = re.sub(r"\s+", " ", sentence).strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        found.append(clean)
    return found


def with_amalgamate_follow_evidence(
    items: list[EvidenceItem], text: str
) -> tuple[list[EvidenceItem], bool]:
    """融合题补上完整的整体句。截断引文不能代替整句。"""
    sentences = amalgamate_follow_sentences(text)
    if not sentences:
        return list(items), False
    out = list(items)
    existing = " ".join((item.quote or "") for item in out).lower()
    shown = False
    for index, sentence in enumerate(sentences, start=1):
        if sentence.lower() in existing:
            shown = True
            continue
        out.append(
            EvidenceItem(
                id=f"amalgamate-follow-{index}",
                topic="modify_adapt",
                quote=sentence,
                cue="amalgamate_follows_modify",
                source="regex",
            )
        )
        shown = True
    return out, shown


def _copyright_grant_blob(text: str) -> str:
    """版权授予单元拼接；去掉 AUP/Attachment 段，避免「不得 use」进授权。

    CC 等把 reproduce/Share 写在 hereby grants 之后的列举项，需并入扫描窗。
    """
    units = list(split_units(text or ""))
    parts: list[str] = []
    for i, (_s, _e, unit) in enumerate(units):
        if _unit_kind(unit) != "grant":
            continue
        if _AUP_SEGMENT_RE.search(unit):
            continue
        if _NO_RIGHTS_GRANT_RE.search(unit):
            continue
        parts.append(unit)
        for j in range(i + 1, min(i + 6, len(units))):
            nxt = units[j][2]
            if _unit_kind(nxt) in {"grant", "patent"}:
                break
            if _AUP_SEGMENT_RE.search(nxt):
                break
            if re.search(
                r"reproduce|Share|Adapted Material|\bmodify\b|\bdistribute\b|"
                r"\bsublicense\b|\bmerge\b",
                nxt,
                re.IGNORECASE,
            ):
                parts.append(nxt)
            elif len(nxt) > 300:
                break
            else:
                # 短节标题（Term / Downstream）则停止
                if re.match(r"^\s*\d+\.\s+\w+", nxt):
                    break
    return " ".join(parts)


def _grant_verb_flags(grant_blob: str) -> dict[str, bool]:
    if not grant_blob:
        return {
            "grant_use": False,
            "grant_modify": False,
            "grant_redistribute": False,
            "grant_sublicense": False,
            "grant_merge": False,
        }
    return {
        "grant_use": bool(_GRANT_USE_RE.search(grant_blob)),
        "grant_modify": bool(_GRANT_MODIFY_RE.search(grant_blob)),
        "grant_redistribute": bool(_GRANT_REDISTRIBUTE_RE.search(grant_blob)),
        "grant_sublicense": bool(_GRANT_SUBLICENSE_RE.search(grant_blob)),
        "grant_merge": bool(_GRANT_MERGE_RE.search(grant_blob)),
    }


def _content_data_work(text: str) -> bool:
    """定义句把数据库、数据集、语料或文学艺术作品写成许可对象。"""
    return _any_kept_definition(text, _data_defined_in_sentence)


def _model_work(text: str) -> bool:
    """定义句把权重、checkpoint 或参数写成许可对象。"""
    return _any_kept_definition(text, _model_defined_in_sentence)


_SOURCE_FORM_RE = re.compile(
    r"\bsource(?:\s+code)?\s+form\b|\bin source(?:\s+code)?\b|\bas source\b",
    re.IGNORECASE,
)
_BINARY_FORM_RE = re.compile(
    r"\b(?:object code|executable|compiled|binary)\s+form\b|"
    r"\bin (?:object code|binary)\b",
    re.IGNORECASE,
)
_BOTH_FORMS_RE = re.compile(
    r"source(?:\s+code)?\s+or\s+binary|"
    r"source(?:\s+code)?\s+and\s+binary|"
    r"\bin any form\b",
    re.IGNORECASE,
)
# 再分发的副本须仍按本许可授权。名称可以是 license，也可以是 this 后面的本许可专名。
_COPIES_SAME_LICENSE_RE = re.compile(
    r"\b(?:shall|must)\s+be\s+licensed\s+under\s+(?:the\s+terms\s+of\s+)?this\b"
    r"(?:\s+[A-Za-z0-9]+){0,6}\s+licen[cs]es?\b",
    re.IGNORECASE,
)
_DISTRIBUTE_VERB_RE = re.compile(
    r"\b(?:distribut|communicat|convey)\w*\b",
    re.IGNORECASE,
)
_WORK_OR_COPY_RE = re.compile(
    r"\b(?:works?|derivativ\w*|copies|copy)\b",
    re.IGNORECASE,
)
_MODIFY_LICENSE_TEXT_RE = re.compile(
    r"\bmodify this (?:licen[cs]e|agreement)\b",
    re.IGNORECASE,
)
_GOVERNED_RE = re.compile(
    r"only under this (?:licen[cs]e|agreement)|"
    r"under the terms of this (?:licen[cs]e|agreement)|"
    r"made available under this (?:licen[cs]e|agreement)|"
    r"(?:this|which|that) (?:licen[cs]e|agreement) will govern|"
    r"governed by this (?:licen[cs]e|agreement)|"
    r"must be distributed entirely under this (?:licen[cs]e|agreement)|"
    r"must not be distributed under any other (?:licen[cs]e|agreement)|"
    r"must not be released under any other (?:type of )?(?:licen[cs]e|agreement)",
    re.IGNORECASE,
)
_WHOLE_WORK_RE = re.compile(
    r"\bentire work\b|\bas a whole\b|"
    r"distributed entirely under this|"
    r"under any other (?:licen[cs]e|agreement)",
    re.IGNORECASE,
)
_FILE_OBJECT_RE = re.compile(
    r"\bany file\b.{0,80}\bcontains\b|\bfiles? that contain\b",
    re.IGNORECASE | re.DOTALL,
)
_ATTACH_SATISFIES_RE = re.compile(
    r"only under this (?:licen[cs]e|agreement).{0,120}"
    r"(?:by |which you satisfy by )?(?:including|attaching|providing) "
    r"(?:a |the )?(?:complete )?copy of this (?:licen[cs]e|agreement)",
    re.IGNORECASE | re.DOTALL,
)
_OTHER_TERMS_RE = re.compile(
    r"\b(?:different|other)\s+terms\b|"
    r"\banother licen[cs]e\b|"
    r"\ba licen[cs]e other than\b|"
    r"\bdifferent than this\b|"
    r"\bunder (?:a |any )?licen[cs]e of your choice\b",
    re.IGNORECASE,
)
_COMPLIES_RE = re.compile(
    r"\bcomplies with this (?:licen[cs]e|agreement)\b|"
    r"\blicen[cs]e that complies\b|"
    r"\bmeets the requirements of this (?:licen[cs]e|agreement)\b",
    re.IGNORECASE,
)
_AUTO_GRANT_RE = re.compile(
    r"\bautomatically\b.{0,50}\b(?:receives?|is granted|are granted|be granted)\b.{0,40}\blicen[cs]e\b|"
    r"\brecipient\b.{0,80}\bautomatically\b.{0,50}\b(?:receives?|is granted|are granted|granted)\b.{0,40}\blicen[cs]e\b",
    re.IGNORECASE | re.DOTALL,
)
_DIST_SUBLICENSE_RE = re.compile(
    r"\bsublicense\b.{0,80}\b(?:different|other)\s+terms\b|"
    r"\b(?:executable|object code|binary|compiled)\b.{0,120}\bsublicense\b|"
    r"\bsublicense\b.{0,80}\b(?:executable|object code|binary|compiled)\b",
    re.IGNORECASE | re.DOTALL,
)


def _analyze_distribution(text: str) -> dict[str, object]:
    """按句子作用锁定同许可档位和按形态换证。不使用许可专名。"""
    sentences = iter_license_sentences(text or "")
    whole = False
    file_both = False
    source_governed = False
    attach = False
    binary_free = False
    binary_conditional = False
    for sentence in sentences:
        governed = _GOVERNED_RE.search(sentence) is not None
        source = _SOURCE_FORM_RE.search(sentence) is not None
        binary = _BINARY_FORM_RE.search(sentence) is not None
        both = _BOTH_FORMS_RE.search(sentence) is not None or (source and binary)
        other_terms = _OTHER_TERMS_RE.search(sentence) is not None
        complies = _COMPLIES_RE.search(sentence) is not None
        if governed and _WHOLE_WORK_RE.search(sentence) and not both:
            whole = True
        # 分发本作品或衍生作品的副本时，这些副本须仍受本许可约束：整体同许可。
        # 只约束源码形态的句子留给 source_only，不在这里抬成整体。
        if (
            _COPIES_SAME_LICENSE_RE.search(sentence)
            and _DISTRIBUTE_VERB_RE.search(sentence)
            and _WORK_OR_COPY_RE.search(sentence)
            and not _MODIFY_LICENSE_TEXT_RE.search(sentence)
            and not (source and not binary)
        ):
            whole = True
        if governed and _FILE_OBJECT_RE.search(sentence) and both:
            file_both = True
        if governed and source and not binary:
            source_governed = True
        if _ATTACH_SATISFIES_RE.search(sentence):
            attach = True
            source_governed = True
        # 没把义务限定在源码上的换证句，视为另一种形态。
        other_form = not (source and not binary)
        prohibited = re.search(r"\b(?:must|may|shall)\s+not\b", sentence, re.IGNORECASE)
        if other_form and other_terms and not prohibited:
            binary_free = True
        elif other_form and complies and not other_terms and not prohibited:
            binary_conditional = True
    if binary_free:
        binary_conditional = False
    if file_both:
        scope = "file_all_forms"
    elif source_governed and (binary_free or binary_conditional):
        scope = "source_only"
    elif whole and not (binary_free or binary_conditional):
        scope = "whole"
    else:
        scope = "none"
    return {
        "same_license_scope": scope,
        "source_same_by_attaching": bool(attach and scope == "source_only"),
        "binary_relicense_free": bool(binary_free and scope == "source_only"),
        "binary_relicense_conditional": bool(
            binary_conditional and scope == "source_only"
        ),
        "auto_downstream_grant": bool(_AUTO_GRANT_RE.search(text or "")),
    }


_DISTILL_NAMED_RE = re.compile(
    r"\b(?:will not|shall not|may not|must not|agree not to)\b"
    r"(?!\s+(?:apply|include|be liable)\b)"
    r".{0,500}?"
    r"(?:derivative works?\b|derivatives of the (?:model|software|work)|"
    r"model derivatives|data produced by the software|\boutputs?\b)",
    re.IGNORECASE | re.DOTALL,
)
_IMPROVE_OTHER_MODEL_RE = re.compile(
    r"improve any other",
    re.IGNORECASE,
)


_DISTILL_OBJECT_RE = re.compile(
    r"derivative works?\b|derivatives of the (?:model|software|work)|"
    r"model derivatives|data produced by the software|\boutputs?\b",
    re.IGNORECASE,
)
_DISTILL_LEAD_RE = re.compile(
    r"\b(?:will not|shall not|may not|must not|agree not to)\b"
    r"(?!\s+(?:apply|include|be liable)\b).{0,120}$",
    re.IGNORECASE | re.DOTALL,
)


def distill_named_restriction(body: str) -> bool:
    """限制句点名衍生、输出或软件产出的数据。只遵守另一份使用政策不算。"""
    sentences = iter_license_sentences(body or "")
    for index, sentence in enumerate(sentences):
        if _IMPROVE_OTHER_MODEL_RE.search(sentence):
            continue
        if _DISTILL_NAMED_RE.search(sentence):
            return True
        nxt = sentences[index + 1] if index + 1 < len(sentences) else ""
        if (
            nxt
            and _DISTILL_LEAD_RE.search(sentence)
            and not _IMPROVE_OTHER_MODEL_RE.search(nxt)
            and _DISTILL_OBJECT_RE.search(nxt)
        ):
            return True
    return False


_PURPOSE_LIMIT_RE = re.compile(
    r"shall only use\b.{0,280}for research purposes|"
    r"for non-?commercial research purposes only|"
    r"solely for\b.{0,100}non-?commercial research",
    re.IGNORECASE | re.DOTALL,
)
_PURPOSE_PASSON_RE = re.compile(
    r"derivatives?.{0,220}subject to the restrictions|"
    r"additional or different licen[cs]e terms.{0,320}"
    r"limit the use\b.{0,220}to non-?commercial research purposes",
    re.IGNORECASE | re.DOTALL,
)


def purpose_use_passes_on(body: str) -> bool:
    """用途限制，并且衍生或另附条款仍须维持该用途。两句都要有。"""
    text = body or ""
    return bool(_PURPOSE_LIMIT_RE.search(text) and _PURPOSE_PASSON_RE.search(text))


_DERIVATIVE_STATE_CHANGES_RE = re.compile(
    r"(?:derivatives|derivative works).{0,180}"
    r"prominent notices stating that you modified|"
    r"prominent notices stating that you modified.{0,80}"
    r"(?:derivatives|derivative works)",
    re.IGNORECASE | re.DOTALL,
)
_AGGREGATION_EXCLUDED_RE = re.compile(
    r"mere aggregation|"
    r"aggregat\w*.{0,160}(?:does not|do not|is not|are not).{0,80}"
    r"(?:under the scope|derivative|based on)|"
    r"(?:collection|compilation)s?.{0,100}"
    r"(?:is not|are not|does not constitute).{0,40}derivative",
    re.IGNORECASE | re.DOTALL,
)


def derivative_must_state_changes(body: str) -> bool:
    """同一句里，分发的衍生必须标明修改。

    把集合或聚合排除出衍生的那一句本身不算。另一句仍要求衍生标明修改时，线索为真。
    """
    for sentence in re.split(r"(?<=[.!?])\s+", body or ""):
        if _AGGREGATION_EXCLUDED_RE.search(sentence):
            continue
        if _DERIVATIVE_STATE_CHANGES_RE.search(sentence):
            return True
    return False


_TRANSLATION_IN_MODIFICATION_RE = re.compile(
    r"translation is\s+included without limitation in\s+the term\s+[\"']?modification",
    re.IGNORECASE,
)
_SEPARATE_TRANSLATION_RE = re.compile(
    r"adapted material.{0,200}\btranslat|"
    r"\btranslat\w*.{0,80}adapted material",
    re.IGNORECASE,
)
_PROGRAM_SUBJECT_RE = re.compile(
    r"\b(?:the program|the library|software library|this program)\b",
    re.IGNORECASE,
)


def embed_inherits_modify(body: str) -> bool:
    """翻译被算进修改，且这句说的不是程序或库时，嵌入继承修改。"""
    for sentence in iter_license_sentences(body or ""):
        if not _TRANSLATION_IN_MODIFICATION_RE.search(sentence):
            continue
        if _PROGRAM_SUBJECT_RE.search(sentence):
            return False
        if _any_kept_definition(body, _software_defined_in_sentence):
            return False
        flat = re.sub(r"\s+", " ", body or "")
        if _SEPARATE_TRANSLATION_RE.search(flat):
            return False
        return True
    return False


_WHOLE_THIS_LICENSE_RE = re.compile(
    r"distribution of the whole must be on the terms of this license",
    re.IGNORECASE,
)
_TERMS_OF_CHOICE_RE = re.compile(r"under terms of your choice", re.IGNORECASE)
_OTHER_SIDE_PROGRAM_RE = re.compile(r"\ba program\b", re.IGNORECASE)


def other_kind_exception(body: str) -> bool:
    """许可对象不是软件，而例外点名程序时，记到混合类型组合。库与应用程序不算。"""
    if _any_kept_definition(body, _software_defined_in_sentence):
        return False
    flat = re.sub(r"\s+", " ", body or "")
    if not _WHOLE_THIS_LICENSE_RE.search(flat):
        return False
    if not _TERMS_OF_CHOICE_RE.search(flat):
        return False
    return _OTHER_SIDE_PROGRAM_RE.search(flat) is not None


_YOUR_DERIV_TERMS_RE = re.compile(
    r"derivatives made by or for you.{0,120}additional or different terms",
    re.IGNORECASE | re.DOTALL,
)
_LICENSOR_DERIV_STAYS_RE = re.compile(
    r"derivatives made by or for (?!you\b)\w.{0,160}remain governed by the terms",
    re.IGNORECASE | re.DOTALL,
)
def your_derivative_may_relicense(body: str) -> bool:
    """两支都要有：许可方自己的衍生仍受本协议约束，你做的衍生可以另附条款。"""
    flat = re.sub(r"\s+", " ", body or "")
    return bool(
        _LICENSOR_DERIV_STAYS_RE.search(flat) and _YOUR_DERIV_TERMS_RE.search(flat)
    )


def attach_license_cue(body: str) -> bool:
    """向接收方提供本协议或本许可的副本。另签协议、以及「附上并不足以换许可」那句不算。"""
    for sentence in iter_license_sentences(body or ""):
        if _NOT_SUFFICIENT_ATTACH_RE.search(sentence):
            continue
        if _ATTACH_LICENSE_RE.search(sentence):
            return True
    return False


def extract_cues(text: str) -> TextCues:
    """全文套话：权利前缀从版权授权句和专利授权句抬；源码交付渠道不升成总前缀。"""
    body = text or ""
    prefixes: list[str] = []
    seen: set[str] = set()
    non_sublicensable = next(p for k, p in _PREFIX_CUES if k == "non-sublicensable")
    has_non_sublicensable = bool(non_sublicensable.search(body))
    grant_blob = " ".join(
        unit
        for _s, _e, unit in split_units(body)
        if _unit_kind(unit) in {"grant", "patent"}
    )
    copyright_blob = _copyright_grant_blob(body)
    verb_flags = _grant_verb_flags(copyright_blob)
    scan = grant_blob or ""
    for key, pattern in _PREFIX_CUES:
        if key == "sublicensable" and has_non_sublicensable:
            continue
        if key == "no-charge" and not scan:
            continue
        target = scan if key in {
            "perpetual",
            "worldwide",
            "non-exclusive",
            "no-charge",
            "royalty-free",
            "irrevocable",
            "non-transferable",
            "revocable",
            "sublicensable",
            "any-person",
            "non-sublicensable",
        } else body
        if not pattern.search(target):
            continue
        if key in seen:
            continue
        seen.add(key)
        prefixes.append(key)
    clean_yes, follows_modify = different_terms_flags(body)
    patent_grant, patent_denied = _patent_flags(body)
    distribution = _analyze_distribution(body)
    granted_sub = bool(_SUBLICENSE_GRANT_RE.search(copyright_blob)) and not bool(
        _SUBLICENSE_BAN_RE.search(body)
    )
    dist_sub = bool(_DIST_SUBLICENSE_RE.search(body)) and not granted_sub
    return TextCues(
        rights_prefix=prefixes,
        non_commercial=text_has_non_commercial(body),
        patent_grant=patent_grant,
        patent_denied=patent_denied,
        trademark_use_denied=_any_sentence(body, _trademark_use_in_sentence),
        trademark_right_reserved=bool(_TRADEMARK_RIGHT_RESERVED_RE.search(body)),
        software_work=_any_kept_definition(body, _software_defined_in_sentence),
        nd_no_redistribute=_any_sentence(body, _nd_in_sentence),
        commercial_use_limited=_any_sentence(body, _commercial_use_limited_in_sentence),
        sole_source_downstream=_any_sentence(body, _sole_source_in_sentence),
        waive_copyright=_copyright_dedicated(body),
        sublicense_forbidden=bool(_SUBLICENSE_BAN_RE.search(body)),
        sublicense_granted=granted_sub,
        mark_changes=bool(_MARK_CHANGES_RE.search(body) or _DESCRIBE_CHANGES_RE.search(body)),
        attach_license=attach_license_cue(body),
        retain_notices=bool(_RETAIN_NOTICE_RE.search(body)),
        binary_skips_attach_license=_binary_skips_attach_license(body),
        different_terms_relicense=clean_yes,
        amalgamate_follows_modify=follows_modify,
        content_data_work=_content_data_work(body),
        model_work=_model_work(body),
        ultra_permissive=bool(_ULTRA_PERMISSIVE_RE.search(body)),
        same_license_scope=str(distribution["same_license_scope"]),
        source_same_by_attaching=bool(distribution["source_same_by_attaching"]),
        binary_relicense_free=bool(distribution["binary_relicense_free"]),
        binary_relicense_conditional=bool(distribution["binary_relicense_conditional"]),
        auto_downstream_grant=bool(distribution["auto_downstream_grant"]),
        sublicense_distribution_only=dist_sub,
        distill_named_restriction=distill_named_restriction(body),
        purpose_use_passes_on=purpose_use_passes_on(body),
        derivative_must_state_changes=derivative_must_state_changes(body),
        embed_inherits_modify=embed_inherits_modify(body),
        other_kind_exception=other_kind_exception(body),
        your_derivative_may_relicense=your_derivative_may_relicense(body),
        **verb_flags,
    )


def _binary_skips_attach_license(body: str) -> bool:
    """源码须附完整许可文本，且存在目标码/binary 分句但未要求附许可文本。"""
    text = body or ""
    if not _SOURCE_ATTACH_LICENSE_RE.search(text):
        return False
    if not _BINARY_FORM_CLAUSE_RE.search(text):
        return False
    if _BINARY_ATTACH_LICENSE_RE.search(text):
        return False
    return True


def _patent_flags(body: str) -> tuple[bool, bool]:
    """正向专利授予 vs 真不授予；限定性「other than licensed claims」不算 deny。"""
    text = body or ""
    grant = bool(_PATENT_GRANT_RE.search(text))
    denied = False
    for match in _PATENT_DENIED_RE.finditer(text):
        window = text[max(0, match.start() - 40) : match.end() + 120]
        if _PATENT_DENY_SCOPED_RE.search(window):
            continue
        denied = True
        break
    # 有本许可正向授予时，不以限定排除句盖掉 grant
    if grant:
        denied = False
    return grant, denied


_ADVICE_NOT_CONVERT_RE = re.compile(
    r"if this is what you want to do|you may consider it more useful",
    re.IGNORECASE,
)
_INCOMPAT_NAMED_RE = re.compile(
    r"incompatible with the gnu|not compatible with the gnu|"
    r"incompatible with (?:the )?(?:gpl|agpl|lgpl|gfdl)\b",
    re.IGNORECASE,
)
_INCOMPAT_AIM_RE = re.compile(r"incompatible with the aim", re.IGNORECASE)
_VERSION_NEAR_RE = re.compile(r"version\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_GPL_ALIAS_RE = re.compile(
    r'["“]GNU GPL["”] refers to version (\d+(?:\.\d+)?) of the GNU\s+General Public License',
    re.IGNORECASE,
)
_REJECT_GOVERNED_RE = re.compile(r"remain governed by", re.IGNORECASE)
_APPLY_COMBINATION_RE = re.compile(
    r"special requirements of\b.{0,500}?apply to the\s+combination",
    re.IGNORECASE | re.DOTALL,
)
_OPT_APPLY_RE = re.compile(r"opt to apply the terms of", re.IGNORECASE)
_CONVEY_UNDER_GPL_RE = re.compile(
    r"under the gnu gpl,\s+with none of the additional permissions",
    re.IGNORECASE,
)
_UNDECIDED_ADOPT_RE = re.compile(
    r"you may (?:also )?license\b.{0,300}?\bunder\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class NamedMention:
    sid: str
    quote: str
    rule: str


@dataclass
class NamedClassification:
    allow: list[str]
    ban: list[str]
    rejected: list[str]
    mentions: list[NamedMention]


def _version_in(blob: str) -> Optional[str]:
    m = _VERSION_NEAR_RE.search(blob or "")
    return m.group(1) if m else None


def _short_id_for_title(blob: str) -> Optional[str]:
    """只把引文里的法定全称映射成 short_id，不做兼容判决。"""
    lower = (blob or "").lower()
    ver = _version_in(lower)
    if "affero general public license" in lower:
        if ver in {None, "3", "3.0"}:
            return "AGPL-3.0"
        if ver in {"1", "1.0"}:
            return "AGPL-1.0"
        return None
    if "lesser general public license" in lower or "library general public license" in lower:
        if ver in {"3", "3.0"}:
            return "LGPL-3.0"
        if ver in {"2.1"}:
            return "LGPL-2.1"
        if ver in {"2", "2.0"}:
            return "LGPL-2.0"
        return None
    if "gnu general public license" in lower or "ordinary gnu general public license" in lower:
        if ver in {"3", "3.0"}:
            return "GPL-3.0"
        if ver in {"2", "2.0"}:
            return "GPL-2.0"
        if ver in {"1", "1.0"}:
            return "GPL-1.0"
        return None
    return None


def normalize_named_license(raw: str) -> Optional[str]:
    """把模型抄出的称呼或已是短名的字符串归一。不接受未在别名表中的猜测。"""
    text = (raw or "").strip()
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text)
    lower = compact.lower()
    known = {
        "agpl-3.0": "AGPL-3.0",
        "agpl-3": "AGPL-3.0",
        "agpl-1.0": "AGPL-1.0",
        "gpl-3.0": "GPL-3.0",
        "gpl-3": "GPL-3.0",
        "gpl-2.0": "GPL-2.0",
        "gpl-2": "GPL-2.0",
        "gpl-1.0": "GPL-1.0",
        "lgpl-3.0": "LGPL-3.0",
        "lgpl-2.1": "LGPL-2.1",
        "lgpl-2.0": "LGPL-2.0",
    }
    if lower in known:
        return known[lower]
    return _short_id_for_title(compact)


def _filter_self(ids: list[str], self_id: str) -> list[str]:
    me = (self_id or "").strip().lower()
    out: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if not item:
            continue
        key = item.lower()
        if me and key == me:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _gpl_alias(text: str) -> Optional[str]:
    match = _GPL_ALIAS_RE.search(text or "")
    if not match:
        return None
    ver = match.group(1)
    if ver in {"3", "3.0"}:
        return "GPL-3.0"
    if ver in {"2", "2.0"}:
        return "GPL-2.0"
    return None


def _mention_rule(unit: str) -> Optional[str]:
    """新作品整体能否采用本段点名的许可。前言和附上许可文本不算。"""
    if _ADVICE_NOT_CONVERT_RE.search(unit):
        return None
    if _REJECT_GOVERNED_RE.search(unit):
        return "reject"
    if _APPLY_COMBINATION_RE.search(unit) or _OPT_APPLY_RE.search(unit):
        return "allow"
    if _CONVEY_UNDER_GPL_RE.search(unit):
        return "allow"
    if _UNDECIDED_ADOPT_RE.search(unit):
        return "undecided"
    return None


def _sid_for_mention(unit: str, rule: str, alias: Optional[str]) -> Optional[str]:
    if rule == "allow" and _CONVEY_UNDER_GPL_RE.search(unit):
        return alias or _short_id_for_title(unit)
    return _short_id_for_title(unit)


_DEFINED_TERM_RE = re.compile(
    r"[\"“”']?([A-Z][A-Za-z0-9]+(?:[ \-][A-Za-z0-9]+){0,6})[\"“”']?\s+means\b"
)
_TITLE_SCAN_RE = re.compile(
    r"((?:gnu\s+)?(?:affero|lesser|library)\s+general public license|"
    r"gnu\s+general public license)"
    r"(?:[^.]{0,40}?version\s+(\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_MAY_UNDER_RE = re.compile(
    r"\b(?:may|can|permits?|permitted|allowed)\b.{0,180}"
    r"\b(?:distribute|convey|made available|license)\w*\b.{0,80}\bunder\b",
    re.IGNORECASE | re.DOTALL,
)
_SPECIMEN_RE = re.compile(
    r"\bexhibit\b|\bform of notice\b|\bas defined by\b|\bas described by\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(r"\bif\b", re.IGNORECASE)
_INCOMPAT_SPAN_RE = re.compile(r"\bincompatible with\b(.{0,80})", re.IGNORECASE)


def _term_key(raw: str) -> str:
    text = re.sub(r"^(?:the|a|an)\s+", "", (raw or "").strip(), flags=re.IGNORECASE)
    text = re.sub(r"[\"“”']", "", text)
    return re.sub(r"\s+", " ", text).lower()


def _term_pattern(key: str) -> str:
    """末词允许单复数，避免把 Secondary 收成一个过短的词。"""
    words = [part for part in key.split() if part]
    if not words:
        return r"$^"
    bits: list[str] = []
    for index, word in enumerate(words):
        if index == len(words) - 1 and word.endswith("s"):
            bits.append(re.escape(word[:-1]) + r"s?")
        elif index == len(words) - 1:
            bits.append(re.escape(word) + r"s?")
        else:
            bits.append(re.escape(word))
    return r"\b" + r"\s+".join(bits) + r"\b"


def _sids_in_blob(blob: str) -> list[str]:
    found: list[str] = []
    for match in _TITLE_SCAN_RE.finditer(blob or ""):
        title = match.group(1)
        ver = match.group(2) or ""
        sid = _short_id_for_title(f"{title} version {ver}".strip())
        if sid and sid not in found:
            found.append(sid)
    return found


def _license_aliases(text: str) -> dict[str, list[str]]:
    """「某名称 means 许可全称」只建别名，名称本身不进 compat。"""
    aliases: dict[str, list[str]] = {}
    for sentence in iter_license_sentences(text or ""):
        match = _DEFINED_TERM_RE.search(sentence)
        if not match:
            continue
        key = _term_key(match.group(1))
        if len(key) < 3 or key in {"license", "agreement"}:
            continue
        ids = _sids_in_blob(sentence)
        if ids:
            aliases[key] = ids
    return aliases


def _alias_ids_in(blob: str, aliases: dict[str, list[str]]) -> list[str]:
    lower = (blob or "").lower()
    found: list[str] = []
    for key, ids in aliases.items():
        if re.search(_term_pattern(key), lower):
            for sid in ids:
                if sid not in found:
                    found.append(sid)
    return found


def classify_named_licenses(text: str, self_id: str = "") -> NamedClassification:
    """点名句先分成允许、拒绝、未决。未决留给模型判断新作品整体能否采用。"""
    allow: list[str] = []
    ban: list[str] = []
    rejected: list[str] = []
    mentions: list[NamedMention] = []
    alias = _gpl_alias(text or "")
    aliases = _license_aliases(text or "")
    for _s, _e, unit in split_units(text or ""):
        if _INCOMPAT_NAMED_RE.search(unit) and not _INCOMPAT_AIM_RE.search(unit):
            sid = _short_id_for_title(unit)
            if sid:
                ban.append(sid)
            for token, mapped in (
                ("agpl", "AGPL"),
                ("lgpl", "LGPL"),
                ("gfdl", "GFDL"),
                ("gpl", "GPL"),
            ):
                if re.search(rf"\b{token}\b", unit, re.IGNORECASE):
                    ban.append(mapped)
        rule = _mention_rule(unit)
        if not rule:
            continue
        sid = _sid_for_mention(unit, rule, alias)
        if not sid:
            continue
        quote = re.sub(r"\s+", " ", unit).strip()
        mentions.append(NamedMention(sid=sid, quote=quote[:1800], rule=rule))
        if rule == "allow":
            allow.append(sid)
        elif rule == "reject":
            rejected.append(sid)
    for sentence in iter_license_sentences(text or ""):
        operative_ban = (
            _INCOMPAT_SPAN_RE.search(sentence)
            and not _INCOMPAT_AIM_RE.search(sentence)
            and not re.search(r"\bnot\s+incompatible\b", sentence, re.IGNORECASE)
            and not _HYPOTHETICAL_RE.search(sentence)
            and not _SPECIMEN_RE.search(sentence)
            and not _DEFINED_TERM_RE.search(sentence)
        )
        if operative_ban:
            ban.extend(_alias_ids_in(sentence, aliases))
        elif _MAY_UNDER_RE.search(sentence) and not _SPECIMEN_RE.search(sentence):
            allow.extend(_alias_ids_in(sentence, aliases))
    if ban:
        blocked = {item.lower() for item in ban}
        allow = [item for item in allow if item.lower() not in blocked]
    return NamedClassification(
        allow=_filter_self(allow, self_id),
        ban=_filter_self(ban, self_id),
        rejected=_filter_self(rejected, self_id),
        mentions=[
            item
            for item in mentions
            if item.sid.lower() != (self_id or "").strip().lower()
        ],
    )


def extract_named_licenses(text: str, self_id: str = "") -> tuple[list[str], list[str]]:
    """规则已能确定的 compat / incompat。未决候选不在这里放行。"""
    found = classify_named_licenses(text, self_id)
    return found.allow, found.ban


_OPENRAIL_FAMILY_INCOMPAT = ["GPL", "AGPL", "LGPL", "GFDL"]
_OPENRAIL_USE_IN_AGREEMENT_RE = re.compile(
    r"use-?based restrictions|use restrictions",
    re.IGNORECASE,
)
_OPENRAIL_ENFORCEABLE_RE = re.compile(
    r"enforceable provision.{0,240}agreement.{0,240}(?:model or derivatives|derivatives of the model)",
    re.IGNORECASE | re.DOTALL,
)
_OPENRAIL_DIFFERENT_TERMS_RE = re.compile(
    r"additional or different licen[cs]e terms.{0,240}respecting paragraph"
    r".{0,240}derivatives of the model as a whole",
    re.IGNORECASE | re.DOTALL,
)


def openrail_family_incompat(text: str) -> list[str]:
    """两句同时出现才给出 GPL 一族不兼容名单。不按许可名称判断。"""
    body = re.sub(r"\s+", " ", text or "")
    if not _OPENRAIL_USE_IN_AGREEMENT_RE.search(body):
        return []
    if not _OPENRAIL_ENFORCEABLE_RE.search(body):
        return []
    if not _OPENRAIL_DIFFERENT_TERMS_RE.search(body):
        return []
    return list(_OPENRAIL_FAMILY_INCOMPAT)


def extract_identity(text: str, license_name: str) -> IdentityFacts:
    full_name = license_name
    if text and text.strip() and text.strip() != license_name.strip():
        for line in text.strip().splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) < 4 or _HEADING_SKIP.match(line):
                continue
            if len(line) <= 180:
                full_name = line
            break
    version: Optional[str] = None
    url: Optional[str] = None
    body = text or ""
    vm = _VERSION_RE.search(body)
    if vm:
        version = vm.group(1)
    preferred: list[str] = []
    fallback: Optional[str] = None
    for raw in _URL_RE.findall(body):
        candidate = raw.rstrip(").,")
        if re.search(r"example\.com|localhost", candidate, re.IGNORECASE):
            continue
        lower = candidate.lower()
        if any(
            host in lower
            for host in (
                "gnu.org",
                "opensource.org",
                "spdx.org",
                "creativecommons.org",
                "apache.org",
            )
        ):
            preferred.append(candidate)
        elif fallback is None:
            fallback = candidate
    url = preferred[0] if preferred else fallback
    return IdentityFacts(full_name=full_name, version=version, url=url)
