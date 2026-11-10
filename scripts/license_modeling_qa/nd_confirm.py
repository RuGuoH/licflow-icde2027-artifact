"""ND 线索命中后，只把命中的句子交给模型判断。

问的是这一句是否禁止分发改编版、衍生材料或提取出的模型。
否定则清掉线索，装配不再把 coverage 收成只有原件。
这次调用写入问卷调用日志，不写入事实块日志。
"""
from __future__ import annotations

import logging
import re

from .ask import _ask_batch
from .extract import _nd_in_sentence, iter_license_sentences
from .ir import TextCues

_PROMPT = """下面的句子来自同一份许可。判断其中是否至少有一句在禁止分发改编版、衍生材料或提取出的模型。
禁止使用商标、禁止某些用途、要求分发原件时附带声明，都不是这种禁止。
若一句先列出 use、modify、derivative、distribute，后面限定的是商业、军事、监控等用途，这一句是用途限制，回答 false。
不要根据许可名称判断。
只输出 JSON：{"forbids_adaptation_distribution": true 或 false}

"""


def nd_trigger_sentences(license_text: str) -> list[str]:
    found = []
    for sentence in iter_license_sentences(license_text or ""):
        if _nd_in_sentence(sentence):
            found.append(re.sub(r"\s+", " ", sentence).strip())
    return found


def confirm_nd_cue(helper, license_name: str, license_text: str, cues: TextCues) -> TextCues:
    """线索为真时询问命中句。模型否定则清掉 nd_no_redistribute。无法解析时保留线索。"""
    if not cues.nd_no_redistribute:
        return cues
    sentences = nd_trigger_sentences(license_text)
    if not sentences:
        return cues
    quoted = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1))
    parsed = _ask_batch(helper, license_name, "nd_sentence", _PROMPT + quoted)
    if not isinstance(parsed, dict) or "forbids_adaptation_distribution" not in parsed:
        logging.warning("nd sentence check unparsed, keep cue license=%s", license_name)
        return cues
    value = parsed.get("forbids_adaptation_distribution")
    confirmed = value is True or str(value).strip().lower() == "true"
    if not confirmed:
        cues.nd_no_redistribute = False
        logging.info("nd cue cleared by sentence check license=%s", license_name)
    return cues
