"""根据证据挂问卷包。只决定问什么，不写 categories。"""
from __future__ import annotations

from .gazetteer import (
    COMBINE_TOPICS,
    COMPAT_TOPICS,
    DATA_REUSE_TOPICS,
    DISTILL_TOPICS,
    MODIFY_TOPICS,
    NETWORK_TOPICS,
    OUTPUT_TOPICS,
    RESTRICT_TOPICS,
    TRAIN_TOPICS,
)
from .types import EvidenceItem, PackDecision


def attach_packs(
    evidence: list[EvidenceItem],
    text: str = "",
    named_from_text: bool = False,
) -> PackDecision:
    packs = ["P_l0", "P_combine"]
    reasons = ["L0 与 Combine 包为第一期必挂"]
    if any(e.topic in MODIFY_TOPICS for e in evidence):
        packs.append("P_modify")
        reasons.append("抽到修改/同许可线索")
    elif text and _looks_like_modify(text):
        packs.append("P_modify")
        reasons.append("全文含修改线索")
    if any(e.topic in OUTPUT_TOPICS for e in evidence) or (
        text and _looks_like_output(text)
    ) or any(e.topic in DISTILL_TOPICS for e in evidence):
        packs.append("P_output")
        reasons.append("抽到运行输出或蒸馏线索")
    if not named_from_text and (
        any(e.topic in COMPAT_TOPICS for e in evidence) or _looks_like_convert(text)
    ):
        packs.append("P_compat")
        reasons.append("有点名转换线索但全文抽取未得到 short_id")
    if any(e.topic in RESTRICT_TOPICS for e in evidence):
        packs.append("P_restrict")
        reasons.append("抽到使用限制/AUP 线索")
    if any(e.topic in NETWORK_TOPICS for e in evidence):
        packs.append("P_network")
        reasons.append("抽到远程网络交互线索")
    if any(e.topic in TRAIN_TOPICS for e in evidence):
        packs.append("P_train")
        reasons.append("抽到训练/微调线索")
    if any(e.topic in DATA_REUSE_TOPICS for e in evidence):
        packs.append("P_embed")
        reasons.append("抽到嵌入/增广/统计线索")
    evid_ids = [
        e.id
        for e in evidence
        if e.topic in COMBINE_TOPICS
        or e.topic in MODIFY_TOPICS
        or e.topic in OUTPUT_TOPICS
        or e.topic in DISTILL_TOPICS
        or e.topic in COMPAT_TOPICS
        or e.topic in RESTRICT_TOPICS
        or e.topic in NETWORK_TOPICS
        or e.topic in TRAIN_TOPICS
        or e.topic in DATA_REUSE_TOPICS
    ]
    return PackDecision(packs=packs, evidence_ids=evid_ids, reason="; ".join(reasons))


def _looks_like_modify(text: str) -> bool:
    lower = (text or "").lower()
    return any(k in lower for k in ("modif", "derivative", "adapt", "修改", "衍生"))


def _looks_like_output(text: str) -> bool:
    lower = (text or "").lower()
    return "output from running" in lower or "constitutes a covered work" in lower


def _looks_like_convert(text: str) -> bool:
    lower = (text or "").lower()
    return "opt to apply the terms of" in lower or "gnu affero general public license" in lower
