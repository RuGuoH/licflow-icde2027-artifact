"""第 4.3、4.5、4.6、5 步的单题判断。未接入正式建模。"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .extract import iter_license_sentences

_PURPOSE_RE = re.compile(
    r"research purposes|non-?commercial research|noncommercial research|"
    r"\bnon-?commercial\b|\bnoncommercial\b|"
    r"subject to the restrictions|"
    r"limit the use\b",
    re.IGNORECASE,
)
_TRANSLATION_RE = re.compile(
    r"\btranslations?\b.{0,160}\b(?:modif|derivativ|adapt|included)\b|"
    r"\b(?:modif|derivativ|adapt).{0,160}\btranslations?\b|"
    r"linguistic resources?",
    re.IGNORECASE,
)
_EXCEPTION_RE = re.compile(
    r"terms of your choice|work that uses",
    re.IGNORECASE,
)
_BRANCH_RE = re.compile(
    r"additional or different (?:licen[cs]e )?terms|made by or for",
    re.IGNORECASE,
)

_PURPOSE_PROMPT = """下面的句子来自同一份许可。判断两件事是否同时成立：
1. 使用被限制在研究或非商用研究。
2. 衍生作品必须受这些用途限制，或者接收方另附的条款本身必须把使用限在该用途内。
只限制你自己的使用，回答 false。
只说分发后仍受本协议约束，回答 false。
只说不得附加进一步限制，回答 false。
本许可自己的非商用条件，回答 false。
不要根据许可名称判断。
只输出 JSON：{"purpose_passes_on": true 或 false}

"""
_TRANSLATION_PROMPT = """下面的句子来自同一份许可。判断翻译是否被算进对许可对象本身的修改，并且许可对象就是被翻译的材料。
普通版权许可只是把翻译列成改编的一种，回答 false。
程序许可里提到翻译，回答 false。
不要根据许可名称判断。
只输出 JSON：{"translation_is_modification_of_subject": true 或 false}

"""
_EXCEPTION_PROMPT = """下面的句子来自同一份许可。判断例外点名的是哪一种。
other_kind：作品整体必须使用本许可，而例外点名的是另一类作品，该类作品可以使用其它条款。
library_application：例外是库与使用该库的程序可以按选择的条款组合。
none：没有这两种例外。
不要根据许可名称判断。
只输出 JSON：{"exception_kind": "other_kind" 或 "library_application" 或 "none"}

"""
_BRANCH_PROMPT = """下面的句子来自同一份许可。判断分发衍生是否分成两支：
许可方自己做出的衍生仍受本协议约束，并且你做出的衍生可以另附条款。
两支都有才回答 true。
只允许修改版另附条款，没有「许可方自己的衍生仍受约束」这一支，回答 false。
不要根据许可名称判断。
只输出 JSON：{"your_derivative_other_terms": true 或 false}

"""


def _clean(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence or "").strip()


def _pick(text: str, pattern: re.Pattern[str], *, limit: int = 8) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for sentence in iter_license_sentences(text or ""):
        if not pattern.search(sentence):
            continue
        cleaned = _clean(sentence)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        found.append(cleaned[:900])
        if len(found) >= limit:
            break
    return found


def purpose_sentences(text: str) -> list[str]:
    return _pick(text, _PURPOSE_RE)


def translation_sentences(text: str) -> list[str]:
    return _pick(text, _TRANSLATION_RE)


def exception_sentences(text: str) -> list[str]:
    return _pick(text, _EXCEPTION_RE)


def branch_sentences(text: str) -> list[str]:
    return _pick(text, _BRANCH_RE)


def _ask(helper, license_name: str, label: str, prompt: str, sentences: list[str]) -> Optional[dict]:
    from .ask import _ask_batch

    if not sentences:
        return None
    quoted = "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences, 1))
    parsed = _ask_batch(helper, license_name, label, prompt + quoted)
    return parsed if isinstance(parsed, dict) else None


def _flag(payload: Optional[dict], key: str) -> Optional[bool]:
    if not payload or key not in payload:
        return None
    value = payload.get(key)
    if value is True or str(value).strip().lower() == "true":
        return True
    if value is False or str(value).strip().lower() == "false":
        return False
    return None


def ask_purpose(helper, license_name: str, text: str) -> Optional[bool]:
    return _flag(
        _ask(helper, license_name, "purpose_passes_on", _PURPOSE_PROMPT, purpose_sentences(text)),
        "purpose_passes_on",
    )


def ask_translation(helper, license_name: str, text: str) -> Optional[bool]:
    return _flag(
        _ask(
            helper,
            license_name,
            "translation_subject",
            _TRANSLATION_PROMPT,
            translation_sentences(text),
        ),
        "translation_is_modification_of_subject",
    )


def ask_exception(helper, license_name: str, text: str) -> Optional[str]:
    payload = _ask(
        helper, license_name, "other_kind_exception", _EXCEPTION_PROMPT, exception_sentences(text)
    )
    if not payload:
        return None
    value = str(payload.get("exception_kind") or "").strip().lower()
    if value in {"other_kind", "library_application", "none"}:
        return value
    return None


def ask_branch(helper, license_name: str, text: str) -> Optional[bool]:
    return _flag(
        _ask(helper, license_name, "your_derivative_branch", _BRANCH_PROMPT, branch_sentences(text)),
        "your_derivative_other_terms",
    )


SELECTORS: dict[str, Callable[[str], list[str]]] = {
    "purpose": purpose_sentences,
    "translation": translation_sentences,
    "exception": exception_sentences,
    "branch": branch_sentences,
}
