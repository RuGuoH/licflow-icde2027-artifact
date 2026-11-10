"""主题词表。只做召回，不含 SPDX / 族名规则。"""
from __future__ import annotations

import re
from typing import Pattern

# (topic, regex, cue label)
_RAW_PATTERNS: list[tuple[str, str, str]] = [
    (
        "grant",
        r"permission is hereby granted|\byou may\b|\bgrant(?:s|ed)?\b|\blicensed?\b|"
        r"授予|许可使用",
        "grant",
    ),
    (
        "redistribute",
        r"\bredistribut(?:e|ion|ing)\b|\bdistribut(?:e|ion|ing)\b|\bconvey(?:s|ed|ing)?\b|"
        r"\bshare\b|再分发|分发",
        "distribute",
    ),
    (
        "source_disclose",
        r"source code|corresponding source|complete source|machine-readable source|"
        r"Source Code Form|Executable Form|"
        r"made available as Source Code|"
        r"源代码|完整源",
        "source code",
    ),
    (
        "copyleft_sharealike",
        r"same terms|same license|share\s*alike|sharealike|\bcopyleft\b|"
        r"under this (?:license|licence)|同样许可|相同许可|相同条款",
        "same terms",
    ),
    (
        "modify_adapt",
        r"\bmodif(?:y|ied|ication|ying)\b|\badapt(?:ed|ation)?\b|derivative work|"
        r"based on (?:the |this )?(?:work|program|library)|修改|改编|衍生",
        "modify",
    ),
    (
        "combine_other_uses",
        r"work that uses|combined work|larger work|"
        r"\bapplication\b.{0,40}\blibrary\b|\blibrary\b.{0,40}\bapplication\b|"
        r"terms of your choice|interface definition|header files?|"
        r"shared library|relink(?:ing)?|"
        r"license the entire work|regardless of how they are packaged",
        "work that uses",
    ),
    (
        "combine_agg",
        r"mere aggregation|\baggregat(?:e|ion|ing)\b|"
        r"collection of (?:independent|separate)|"
        r"side-?by-?side|"
        r"\bseparate(?:ly)?\b.{0,40}\bindependent (?:work|works|section)s?\b|"
        r"volume of a (?:storage|distribution)|单纯聚合|仅仅聚合",
        "aggregation",
    ),
    (
        "combine_absorbed",
        r"entire work|combined work as a whole|"
        r"license the work as a whole|"
        r"single (?:combined )?work|constitutes? a derivative|"
        r"整体|作为一部作品",
        "entire work",
    ),
    (
        "combine_database_assimilate",
        r"substantial portion of (?:the )?(?:contents of )?(?:the )?database|"
        r"include all or a substantial portion of the database contents|"
        r"contents in a database in which You have|"
        r"database in which You have Sui Generis Database Rights."
        r"{0,160}Adapted Material|"
        r"\(but not its individual contents\) is Adapted Material|"
        r"but not its individual contents.{0,40}Adapted Material",
        "database assimilate",
    ),
    (
        "adapt_share_conditions",
        r"Adapter's License|"
        r"must not prevent recipients.{0,100}complying with this Public License|"
        r"Adapter's License You apply must be a Creative Commons|"
        r"ShareAlike\.|"
        r"b\. ShareAlike",
        "adapter license",
    ),
    (
        "combine_mix",
        r"mere aggregation of another work not based on the linguistic resource|"
        r"aggregat.{0,80}(?:models?|algorithms?|datasets?)|"
        r"(?:models?|algorithms?|datasets?).{0,80}aggregat|"
        r"combin(?:e|ed|ing) (?:this work |the (?:model|data) )?with other (?:models?|software|data)|"
        r"work of a different (?:type|kind|nature)",
        "mixed aggregation",
    ),
    (
        "combine_link",
        r"\blink(?:s|ed|ing)?\b|\bstatic(?:ally)? link|\bdynamic(?:ally)? link",
        "link",
    ),
    (
        "exception_or_separate_work",
        r"independent work|separate work|identifiable sections?|"
        r"does not bring the other work|not derived from",
        "separate work",
    ),
    ("patent", r"\bpatents?\b|专利权?", "patent"),
    ("trademark", r"\btrademarks?\b|商标", "trademark"),
    (
        "commercial",
        r"\bcommercial(?:ly)?\b|non-?commercial|商业|非商业",
        "commercial",
    ),
    (
        "notice_attribution",
        r"copyright notice|attribution|\bnotices?\b|give credit|"
        r"provide a copy of this licen[cs]e|"
        r"include a copy of this licen[cs]e|"
        r"give.{0,80}a copy of this licen[cs]e|"
        r"保留声明|署名",
        "notice",
    ),
    (
        "sublicense",
        r"sub-?licen[cs]e|再许可",
        "sublicense",
    ),
    (
        "network_saas",
        r"remote network interaction|"
        r"interacting with it remotely|"
        r"remotely through a (?:computer )?network|"
        r"\bas a service\b|\bSaaS\b|"
        r"hosted service|"
        r"remotely accessible|"
        r"API-based access|"
        r"network copyleft|"
        # 远程向他人提供且视同分发（不单靠专章名）
        r"made available as an application intended for use over a network|"
        r"over a network.{0,120}(?:distribution|distribute|communicate)|"
        r"treat any .{0,80}as a distribution|"
        r"external deployment",
        "network",
    ),
    (
        "output_generate",
        r"output from running|\bthe output\b.{0,80}covered|"
        r"constitutes a covered work|\boutputs?\b|运行.{0,10}产生",
        "output",
    ),
    (
        "compat_convert",
        r"compatib(?:le|ility) with|you may (?:also )?license|"
        r"\bAffero\b|additional permission|"
        r"opt to apply the terms of|ordinary gnu general public license|"
        r"instead of this license",
        "compatible",
    ),
    (
        "train_finetune",
        r"fine-?tun(?:e|ing|ed)|"
        r"finetuning|"
        r"trained model weights|"
        r"training-enabling|"
        r"\b(?:pre)?train(?:ing|ed)?\b.{0,60}(?:model|weights?|llm)|"
        r"(?:model|llm).{0,40}\b(?:pre)?train(?:ing|ed)?\b|"
        r"reparametriz",
        "train",
    ),
    (
        "output_distill",
        r"distill(?:ation|ing)? methods|"
        r"\bdistill(?:ation|ing)?\b|"
        r"improve any other (?:large language )?model|"
        r"transfer of patterns of the weights|"
        r"synthetic data.{0,80}train(?:ing)?",
        "distill",
    ),
    (
        "embed_adapt",
        r"adapted material|"
        r"licensed material is translated|"
        r"\bembed(?:ded|ding)?\b.{0,60}\b(?:data|dataset|corpus|weights)\b|"
        r"\b(?:model|algorithm)\b.{0,40}\bembed(?:ded|ding)?\b|"
        r"\bembed(?:ded|ding)?\b.{0,40}\b(?:using|with|by)\b.{0,40}\b(?:model|algorithm)\b|"
        r"linguistic resource.{0,60}\bembed|"
        r"\bembed.{0,60}linguistic resource",
        "embed",
    ),
    (
        "augment_remix",
        r"\bremix(?:ed|ing)?\b|"
        r"altered, arranged, transformed|"
        r"\baugment(?:ed|ing|ation)?\b|"
        r"otherwise modified.{0,40}adapted material",
        "augment",
    ),
    (
        "stat_compute",
        r"\bstatistical (?:analysis|use|research)\b|"
        r"\bstatistically\b|"
        r"computational use|"
        r"not (?:itself )?a linguistic resource",
        "stat",
    ),
    (
        "use_restriction_aup",
        r"use-based restriction|"
        r"acceptable use|"
        r"prohibited uses|"
        r"restricted uses|"
        r"responsible ai|"
        r"responsibleai|"
        r"use restriction|"
        r"you agree not to use.{0,220}(?:illegal|unlawful|harm minors|prohibited)|"
        r"shall require all of .{0,80}users|"
        # 研究型合同式限制：用途清单 + 限制须随下游传递（结构句，不枚举用途）
        r"commercial or production purposes|"
        r"inconsistent with (?:the )?(?:terms of )?(?:this |the )?(?:licen[cs]e|agreement)|"
        r"offer or impose any terms.{0,100}inconsistent|"
        r"不得用于|使用限制",
        "use restriction",
    ),
    (
        "warranty",
        r"as is|without warranty|warranty of|按[「\"]?原样",
        "warranty",
    ),
    (
        "identity",
        r"\bversion\s+\d|\bhttps?://",
        "identity",
    ),
]

PATTERNS: list[tuple[str, Pattern[str], str]] = [
    (topic, re.compile(pat, re.IGNORECASE | re.DOTALL), cue)
    for topic, pat, cue in _RAW_PATTERNS
]

COMBINE_TOPICS = frozenset(
    {
        "combine_agg",
        "combine_other_uses",
        "combine_absorbed",
        "combine_database_assimilate",
        "combine_mix",
        "combine_link",
        "exception_or_separate_work",
    }
)

MODIFY_TOPICS = frozenset({"modify_adapt", "copyleft_sharealike"})
OUTPUT_TOPICS = frozenset({"output_generate"})
DISTILL_TOPICS = frozenset({"output_distill"})
TRAIN_TOPICS = frozenset({"train_finetune"})
EMBED_TOPICS = frozenset({"embed_adapt"})
AUGMENT_TOPICS = frozenset({"augment_remix"})
STAT_TOPICS = frozenset({"stat_compute"})
DATA_REUSE_TOPICS = EMBED_TOPICS | AUGMENT_TOPICS | STAT_TOPICS
COMPAT_TOPICS = frozenset({"compat_convert"})
RESTRICT_TOPICS = frozenset({"use_restriction_aup"})
NETWORK_TOPICS = frozenset({"network_saas"})

GAP_TOPICS = frozenset(
    COMBINE_TOPICS
    | MODIFY_TOPICS
    | OUTPUT_TOPICS
    | {
        "grant",
        "redistribute",
        "source_disclose",
        "notice_attribution",
        "sublicense",
        "network_saas",
        "use_restriction_aup",
        "train_finetune",
        "output_distill",
        "embed_adapt",
        "augment_remix",
        "stat_compute",
        "adapt_share_conditions",
        "patent",
        "compat_convert",
    }
)
