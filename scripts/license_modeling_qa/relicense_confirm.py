"""只重问修改换证和训练换证，并只改这两行的 relicense。

不重跑整份问卷，不改 result、restrictions、copyleft、合并行。
训练和 distill 捆在同一行时，只把新答案写到拆出的训练行，distill 保持原换证。
修改与 augment、组合等捆在同一行时同样拆开。
原文没有讨论这类产物的换证时，不改已有值。
"""
from __future__ import annotations

import copy
import re
from typing import Any, Optional

from .ask import _ask_batch, _resolve_modify_relicense
from .extract import iter_license_sentences, purpose_use_passes_on
from .types import EvidenceItem

_PROMPT = """下面的句子来自同一份许可。只判断这一类产物再分发时能否改用不同许可。不要回答另一类产物。
按下面顺序，命中第一条就停：
1. 这类产物本身必须用本许可发布，或禁止再分发，回答 no。必须用本许可，指产物要按本许可、作为整体按本许可、或只能按本许可来许可。附上许可文本、保留声明、标明修改，都不是这一条。
2. 允许另用条款，但不得冲突，或使用限制必须写进新条款，或只能换成兼容许可或 Secondary License，回答 conditional。
3. 允许另用条款，且没有第 2 条那种限制，回答 yes。
4. 授予是可再许可的，或无限制并且权利里包含修改和再许可，而分发条件只有附上文本、保留声明、标明修改，回答 yes。
5. 不可再许可，且第 2、3 条都没有明确允许另用条款，回答 no。明确允许另用条款时，不以不可再许可为准。
6. 原文没有讨论这类产物的许可，回答 not_mentioned。配套材料的定义里只是把微调列成脚本用途的，不算讨论了训练产物的许可。
不要根据许可名称判断。
只输出 JSON：{{"relicense": "yes" 或 "no" 或 "conditional" 或 "not_mentioned"}}

这一类产物是：{kind}

"""

_GRANT_RE = re.compile(
    r"hereby grants|permission is hereby granted|sublicensable|"
    r"non-?sublicensable|without restriction",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"additional or different licen|"
    r"under the terms of this|"
    r"as a whole|"
    r"use-based restrictions|"
    r"provide a copy of this|"
    r"prominent notices|"
    r"copyright notice|"
    r"permission notice|"
    r"does not contradict|"
    r"compatible licen|"
    r"Secondary Licen|"
    r"must not be distributed|"
    r"shall not distribute",
    re.IGNORECASE,
)
_MODIFY_CLASS_RE = re.compile(
    r"derivative materials|derivative works|derivatives of the model|"
    r"model derivatives|modified versions?|modifications|adapted material|"
    r"improvements, modifications",
    re.IGNORECASE,
)
_TRAIN_CLASS_RE = re.compile(
    r"\b(?:train(?:ing|ed)?|fine-?tun(?:e|ing|ed)|pretraining)\b",
    re.IGNORECASE,
)
_COMPANION_DEF_RE = re.compile(
    r"(?:complementary|companion) materials",
    re.IGNORECASE,
)
_ALLOWED = {"yes", "no", "conditional", "not_mentioned", "true", "false", "n/a"}


def _clean(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()


def _companion_definition(sentence: str) -> bool:
    return bool(_COMPANION_DEF_RE.search(sentence))


def select_relicense_sentences(license_text: str, kind: str) -> list[str]:
    """附上版权授予句，以及这类产物的分发条件。不改主题词表。"""
    if kind not in {"modify", "train"}:
        raise ValueError(kind)
    picked: list[str] = []
    seen: set[str] = set()

    def add(sentence: str) -> None:
        cleaned = _clean(sentence)
        if not cleaned or cleaned in seen or _companion_definition(cleaned):
            return
        seen.add(cleaned)
        picked.append(cleaned)

    sentences = iter_license_sentences(license_text or "")
    grants = [sentence for sentence in sentences if _GRANT_RE.search(sentence)]
    if kind == "modify":
        classes = [sentence for sentence in sentences if _MODIFY_CLASS_RE.search(sentence)]
    else:
        classes = [
            sentence
            for sentence in sentences
            if _TRAIN_CLASS_RE.search(sentence) or _MODIFY_CLASS_RE.search(sentence)
        ]
    conditions = [sentence for sentence in sentences if _CONDITION_RE.search(sentence)]
    for sentence in grants[:4] + classes[:6] + conditions[:6]:
        add(sentence)
    return picked


def relicense_prompt(kind: str, sentences: list[str]) -> str:
    label = (
        "修改本作品得到的修改版或衍生作品。不要判断训练、蒸馏或从输出迁移得到的模型。"
        if kind == "modify"
        else "训练或微调本模型得到的模型。蒸馏、从输出迁移模式、合成数据得到的模型不要算进这一类。"
        "若训练得到的模型落在原文的修改或衍生定义里，并且没有被排除，就按该定义的分发条件判断。"
    )
    quoted = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1))
    return _PROMPT.format(kind=label) + quoted


def parse_relicense_answer(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict) or "relicense" not in payload:
        return None
    value = payload.get("relicense")
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool):
        value = "yes" if value else "no"
    text = str(value or "").strip().lower()
    if text not in _ALLOWED:
        return None
    if text in {"true", "yes"}:
        return "yes"
    if text in {"false", "no"}:
        return "no"
    if text in {"n/a", "not_mentioned"}:
        return "not_mentioned"
    return "conditional"


def finalize_modify_answer(stated: Optional[str], sentences: list[str], license_text: str) -> Optional[str]:
    """没讨论就不改。讨论了才套用现有的修改锁，以及用途限制向下传递。"""
    if stated in (None, "not_mentioned"):
        return None
    evidence = [
        EvidenceItem(id=f"s{index}", topic="modify_adapt", quote=sentence)
        for index, sentence in enumerate(sentences, 1)
    ]
    locked = _resolve_modify_relicense(stated, "not_mentioned", evidence)
    if locked in {"yes", "true"} and purpose_use_passes_on(license_text or ""):
        return "conditional"
    if locked in {"yes", "true"}:
        return "yes"
    if locked in {"no", "false"}:
        return "no"
    if locked == "conditional":
        return "conditional"
    return None


def finalize_train_answer(stated: Optional[str]) -> Optional[str]:
    if stated in (None, "not_mentioned"):
        return None
    if stated in {"yes", "no", "conditional"}:
        return stated
    return None


def _yaml_relicense(stated: str) -> Any:
    if stated == "yes":
        return True
    if stated == "no":
        return False
    return "conditional"


def _same_relicense(current: Any, stated: str) -> bool:
    target = _yaml_relicense(stated)
    if target is True:
        return current is True
    if target is False:
        return current is False
    return current == "conditional"


def _real_indexes(terms: list[dict], usage: str) -> list[int]:
    found = []
    for index, term in enumerate(terms):
        usages = list(term.get("usages") or [])
        if usage not in usages:
            continue
        if term.get("result") in (None, "", "NODEF") and len(usages) > 1:
            continue
        found.append(index)
    return found


def _write_usage(terms: list[dict], usage: str, stated: str) -> str:
    indexes = _real_indexes(terms, usage)
    if len(indexes) != 1:
        return "skip"
    index = indexes[0]
    if _same_relicense(terms[index].get("relicense"), stated):
        return "unchanged"
    original = copy.deepcopy(terms[index])
    usages = list(original.get("usages") or [])
    target = _yaml_relicense(stated)
    if usages == [usage]:
        terms[index]["relicense"] = target
        return "updated"
    terms[index]["usages"] = [item for item in usages if item != usage]
    keywords = list(terms[index].get("keywords") or [])
    if keywords:
        terms[index]["keywords"] = [item for item in keywords if item != usage]
    split = copy.deepcopy(original)
    split["usages"] = [usage]
    split["relicense"] = target
    if split.get("keywords"):
        own = [item for item in split["keywords"] if item == usage]
        split["keywords"] = own or [usage]
    terms.insert(index + 1, split)
    return "split"


def patch_relicense(
    data: dict,
    *,
    modify: Optional[str] = None,
    train: Optional[str] = None,
) -> tuple[dict, dict[str, str]]:
    """返回新 yaml 和每个用法的写回结果。不改合并行，也不改其它字段。"""
    patched = copy.deepcopy(data)
    terms = list(patched.get("terms") or [])
    notes = {"modify": "absent", "train": "absent"}
    if modify is not None:
        notes["modify"] = _write_usage(terms, "modify", modify)
    if train is not None:
        notes["train"] = _write_usage(terms, "train", train)
    patched["terms"] = terms
    return patched, notes


def ask_relicense(helper, license_name: str, kind: str, license_text: str) -> Optional[str]:
    sentences = select_relicense_sentences(license_text, kind)
    if not sentences:
        return None
    payload = _ask_batch(helper, license_name, f"{kind}_relicense", relicense_prompt(kind, sentences))
    return parse_relicense_answer(payload)
