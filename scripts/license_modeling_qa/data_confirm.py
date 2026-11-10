"""定义句线索为假时，把数据词句交给模型判断许可对象是否包含数据。

线索已经为真时不调用，也不许答案把线索清掉。
false 或解析失败保持线索为假。调用记在问卷日志，标签 work_type_data。
"""
from __future__ import annotations

import logging
import re

from .ask import _ask_batch
from .extract import (
    _DATA_OBJECT_RE,
    _next_sentence_excludes_defined_object,
    iter_license_sentences,
)
from .ir import TextCues

_PROMPT = """下面的句子来自同一份许可。判断其中是否至少有一句在界定许可对象本身包含数据库、数据集、语料或文学艺术作品。
下列都回答 false：权利清单里的 database rights；数据集不属于许可材料；Data 不在本许可下授权；把输出整理成 dataset 时要求附声明。
不要因为 work of authorship 就补 software。
不要根据许可名称判断。
只输出 JSON：{"licensed_object_includes_data": true 或 false}

"""


def data_trigger_sentences(license_text: str) -> list[str]:
    sentences = iter_license_sentences(license_text or "")
    found: list[str] = []
    seen: set[str] = set()

    def add(sentence: str) -> None:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)

    for index, sentence in enumerate(sentences):
        if not _DATA_OBJECT_RE.search(sentence):
            continue
        add(sentence)
        if _next_sentence_excludes_defined_object(sentences, index):
            add(sentences[index + 1])
    return found


def confirm_data_cue(helper, license_name: str, license_text: str, cues: TextCues) -> TextCues:
    """线索为假且出现数据词时询问。true 才补上 content_data_work。"""
    if cues.content_data_work:
        return cues
    sentences = data_trigger_sentences(license_text)
    if not sentences:
        return cues
    quoted = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1))
    parsed = _ask_batch(helper, license_name, "work_type_data", _PROMPT + quoted)
    if not isinstance(parsed, dict) or "licensed_object_includes_data" not in parsed:
        logging.warning("data object check unparsed, keep cue false license=%s", license_name)
        return cues
    value = parsed.get("licensed_object_includes_data")
    if value is True or str(value).strip().lower() == "true":
        cues.content_data_work = True
        logging.info("data cue set by sentence check license=%s", license_name)
    return cues
