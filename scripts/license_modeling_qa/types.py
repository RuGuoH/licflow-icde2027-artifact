"""QA 建模数据契约。抽取 / 问卷 / 装配共用，不依赖 OSADL 或 gold。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Topic = Literal[
    "grant",
    "redistribute",
    "source_disclose",
    "copyleft_sharealike",
    "modify_adapt",
    "exception_or_separate_work",
    "combine_agg",
    "combine_other_uses",
    "combine_absorbed",
    "combine_mix",
    "combine_link",
    "patent",
    "trademark",
    "commercial",
    "notice_attribution",
    "sublicense",
    "network_saas",
    "output_generate",
    "compat_convert",
    "incompat_named",
    "train_finetune",
    "output_distill",
    "embed_adapt",
    "augment_remix",
    "stat_compute",
    "use_restriction_aup",
    "warranty",
    "identity",
]

PackName = Literal[
    "P_l0",
    "P_combine",
    "P_modify",
    "P_output",
    "P_compat",
    "P_restrict",
    "P_network",
    "P_train",
    "P_embed",
]


@dataclass
class EvidenceItem:
    id: str
    topic: str
    quote: str
    start: Optional[int] = None
    end: Optional[int] = None
    cue: str = ""
    # regex = 词表命中；embed = 主题句补召回；fulltext = 短许可整篇
    source: str = "regex"


@dataclass
class PackDecision:
    packs: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Answer:
    question_id: str
    value: str
    evidence_ids: list[str] = field(default_factory=list)
    quote: Optional[str] = None
