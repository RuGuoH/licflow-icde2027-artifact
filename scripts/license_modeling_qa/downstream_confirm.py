"""命中句交给模型判断原许可人是否把本许可下的权利直接给了下游。

没有命中句就不调用，auto_downstream_grant 保持现有正则的结果。
true 写入该线索。false 清掉，即使旧正则曾命中。解析失败保留旧正则的结果。
调用记在问卷日志，标签 downstream_grant，不记成事实块。
"""
from __future__ import annotations

import logging
import re

from .ask import _ask_batch
from .extract import iter_license_sentences
from .ir import TextCues

_PROMPT = """下面的句子来自同一份许可。判断这些句子里是否至少有一句是原许可人把本许可下的权利直接给了下游接收方。
许可人向接收方要约行使许可权利，回答 true。
分发者把自己另行授予的专利许可自动扩给所有接收方，回答 false。
许可终止、自动更新、自动扣费，回答 false。
不要根据许可名称判断。
只输出 JSON：{"direct_downstream_grant": true 或 false}

"""

_PARTY_RE = re.compile(r"\b(?:recipients?|downstream)\b", re.IGNORECASE)
_AUTO_RE = re.compile(r"\bautomatically\b", re.IGNORECASE)
_ACT_RE = re.compile(
    r"\b(?:receiv(?:e|es|ed|ing)|grant(?:s|ed|ing)?|offer(?:s|ed|ing)?|extend(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)


def downstream_trigger_sentences(license_text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for sentence in iter_license_sentences(license_text or ""):
        if not (
            _PARTY_RE.search(sentence)
            and _AUTO_RE.search(sentence)
            and _ACT_RE.search(sentence)
        ):
            continue
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)
    return found


def confirm_downstream_cue(
    helper, license_name: str, license_text: str, cues: TextCues
) -> TextCues:
    """有命中句才询问。true 写入线索，false 清掉，解析失败保持原值。"""
    sentences = downstream_trigger_sentences(license_text)
    if not sentences:
        return cues
    quoted = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1))
    parsed = _ask_batch(helper, license_name, "downstream_grant", _PROMPT + quoted)
    if not isinstance(parsed, dict) or "direct_downstream_grant" not in parsed:
        logging.warning(
            "downstream grant check unparsed, keep cue license=%s", license_name
        )
        return cues
    value = parsed.get("direct_downstream_grant")
    if value is True or str(value).strip().lower() == "true":
        cues.auto_downstream_grant = True
        logging.info("downstream grant set by sentence check license=%s", license_name)
        return cues
    if value is False or str(value).strip().lower() == "false":
        cues.auto_downstream_grant = False
        logging.info("downstream grant cleared by sentence check license=%s", license_name)
        return cues
    logging.warning(
        "downstream grant check unparsed, keep cue license=%s", license_name
    )
    return cues
