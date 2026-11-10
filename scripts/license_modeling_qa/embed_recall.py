"""主题句向量补召回：仅当该 topic 正则缺失且 max_sim≥τ 时写入 EvidenceItem。

不改装配语义；短许可全文模式应跳过本模块。
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Callable, Optional, Sequence

from .embed_client import DEFAULT_EMBED_MODEL, embed_texts
from .ir import TextCues
from .extract import (
    _data_anchor_in_sentence,
    _model_anchor_in_sentence,
    _commercial_use_limited_anchor_in_sentence,
    _nd_anchor_in_sentence,
    _non_commercial_anchor_in_sentence,
    _software_anchor_in_sentence,
    _sole_source_anchor_in_sentence,
    _trademark_use_anchor_in_sentence,
    is_short_fulltext,
    iter_license_sentences,
    split_units,
)
from .topic_sentences import PHASE0_TOPICS, TOPIC_SENTENCES
from .types import EvidenceItem

logger = logging.getLogger(__name__)

# 默认阈值：10 许可标定上 0.55 仍多噪声；0.60 肘点更稳（见 embed_tau_calib.json）。
DEFAULT_EMBED_MIN_SIM = 0.60
# 线索会直接改装配。比问卷补召回更严，避免 0.60 附近的近义句写成硬线索。
CUE_EMBED_MIN_SIM = 0.72
_MAX_QUOTE = 400
EmbedFn = Callable[..., list[list[float]]]


def resolve_embed_min_sim(value: Optional[float] = None) -> float:
    if value is not None:
        return float(value)
    raw = (os.getenv("LICENSE_EMBED_MIN_SIM") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("无效 LICENSE_EMBED_MIN_SIM=%r，使用默认 %.2f", raw, DEFAULT_EMBED_MIN_SIM)
    return DEFAULT_EMBED_MIN_SIM


def embed_supplement_enabled() -> bool:
    """默认开启；LICENSE_EMBED_SUPPLEMENT=0/false 关闭。"""
    raw = (os.getenv("LICENSE_EMBED_SUPPLEMENT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _clip(quote: str) -> str:
    quote = re.sub(r"\s+", " ", quote).strip()
    if len(quote) <= _MAX_QUOTE:
        return quote
    return quote[: _MAX_QUOTE - 1] + "…"


def _avg_vectors(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    n = float(len(vectors))
    return [x / n for x in out]


def score_topics_against_units(
    text: str,
    topics: Sequence[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    embed_fn: Optional[EmbedFn] = None,
) -> dict[str, list[tuple[float, int, int, int, str]]]:
    """对每个 topic 返回 [(sim, unit_index, start, end, unit), ...] 降序。"""
    embed = embed_fn or embed_texts
    units = split_units(text or "")
    if not units:
        return {t: [] for t in topics}

    sentences: list[str] = []
    seen: set[str] = set()
    topic_sent: dict[str, list[str]] = {}
    for topic in topics:
        sents = list(TOPIC_SENTENCES.get(topic) or [])
        topic_sent[topic] = sents
        for s in sents:
            if s not in seen:
                seen.add(s)
                sentences.append(s)

    unit_texts = [u for _a, _b, u in units]
    all_texts = sentences + unit_texts
    vectors = embed(all_texts, model=model)
    sent_vecs = {s: vectors[i] for i, s in enumerate(sentences)}
    unit_vecs = vectors[len(sentences) :]

    out: dict[str, list[tuple[float, int, int, int, str]]] = {}
    for topic in topics:
        sents = topic_sent.get(topic) or []
        if not sents:
            out[topic] = []
            continue
        q = _avg_vectors([sent_vecs[s] for s in sents])
        scored: list[tuple[float, int, int, int, str]] = []
        for idx, ((start, end, unit), vec) in enumerate(zip(units, unit_vecs)):
            scored.append((_cosine(q, vec), idx, start, end, unit))
        scored.sort(key=lambda row: -row[0])
        out[topic] = scored
    return out


def supplement_evidence_with_embed(
    text: str,
    evidence: list[EvidenceItem],
    *,
    min_sim: Optional[float] = None,
    topics: Optional[Sequence[str]] = None,
    model: str = DEFAULT_EMBED_MODEL,
    embed_fn: Optional[EmbedFn] = None,
    per_topic: int = 1,
) -> list[EvidenceItem]:
    """正则缺失的 topic，若 top sim≥τ 则追加 EvidenceItem(source=embed)。

    短许可全文模式直接返回原 evidence。失败时记录日志并返回原列表。
    """
    if is_short_fulltext(text):
        return list(evidence)
    tau = resolve_embed_min_sim(min_sim)
    wanted = list(topics) if topics is not None else list(PHASE0_TOPICS)
    present = {e.topic for e in evidence}
    missing = [t for t in wanted if t not in present and t in TOPIC_SENTENCES]
    # AUP 已在：combine_* 走装配 AUP 补全即可；embed 易捞定义句并误触发 C2。
    # 不要求「无 copyleft_sharealike」——OpenRAIL 等会对定义句误中该 topic。
    _COMBINE_EMBED_SKIP = frozenset(
        {"combine_other_uses", "combine_absorbed", "combine_agg", "combine_link"}
    )
    if "use_restriction_aup" in present:
        missing = [t for t in missing if t not in _COMBINE_EMBED_SKIP]
    if not missing:
        return list(evidence)

    try:
        ranked = score_topics_against_units(
            text, missing, model=model, embed_fn=embed_fn
        )
    except Exception:
        logger.exception("embed 补召回失败，保留正则证据")
        return list(evidence)

    out = list(evidence)
    n = len(out)
    for topic in missing:
        rows = ranked.get(topic) or []
        added = 0
        for sim, _idx, start, end, unit in rows:
            if sim < tau:
                break
            n += 1
            out.append(
                EvidenceItem(
                    id=f"e{n}",
                    topic=topic,
                    quote=_clip(unit),
                    start=start,
                    end=end,
                    cue=f"embed:{sim:.3f}",
                    source="embed",
                )
            )
            added += 1
            if added >= per_topic:
                break
        if added:
            logger.info(
                "embed supplement topic=%s count=%d top_sim=%.3f tau=%.3f",
                topic,
                added,
                rows[0][0] if rows else 0.0,
                tau,
            )
    return out


# 正则没抽到的硬线索。原型是角色说法，不是某一份许可的原句。
_CUE_PROTOTYPES: dict[str, list[str]] = {
    "software_work": [
        "The licensed materials include source code and scripts used to run the model.",
        "Software means the program, source code, and scripts made available under this license.",
    ],
    "non_commercial": [
        "The license is granted only for non-commercial research purposes.",
        "Commercial use is not permitted under this license.",
    ],
    "sole_source_downstream": [
        "Any rights of downstream recipients shall be directly granted by the licensor.",
        "Recipients must obtain a separate license from the licensor and do not receive rights under this license.",
    ],
    "nd_no_redistribute": [
        "You may create derivative works, but you do not have permission to distribute those derivative works.",
        "You may produce adapted material, but not share adapted material.",
    ],
    "commercial_use_limited": [
        "If your monthly active users exceed the stated threshold, you must request a separate license and are not authorized to exercise these rights.",
        "Territory means the worldwide territory, excluding the listed countries.",
    ],
    "trademark_use_denied": [
        "No trademark license is granted, and you may not use the marks to promote or endorse products.",
        "You shall not use the licensor's trademarks for publicity or advertising.",
    ],
    "model_work": [
        "The model means the trained weights, checkpoints, and parameters.",
        "The licensed model consists of learnt weights and checkpoints.",
    ],
    "content_data_work": [
        "The licensed material means a literary or artistic work or a database.",
        "The licensed work is a dataset or corpus.",
    ],
}
_CUE_ANCHORS = {
    "software_work": _software_anchor_in_sentence,
    "non_commercial": _non_commercial_anchor_in_sentence,
    "sole_source_downstream": _sole_source_anchor_in_sentence,
    "nd_no_redistribute": _nd_anchor_in_sentence,
    "commercial_use_limited": _commercial_use_limited_anchor_in_sentence,
    "trademark_use_denied": _trademark_use_anchor_in_sentence,
    "model_work": _model_anchor_in_sentence,
    "content_data_work": _data_anchor_in_sentence,
}


def supplement_cues_with_embed(
    text: str,
    cues: TextCues,
    *,
    min_sim: float = CUE_EMBED_MIN_SIM,
    embed_fn: Optional[EmbedFn] = None,
    model: str = DEFAULT_EMBED_MODEL,
) -> TextCues:
    """正则未命中的线索，用角色原型找一句；过阈值且句内有角色词才写成线索。

    已经为真的线索不会被向量改掉。短文本和接口失败保持原线索。
    """
    if is_short_fulltext(text):
        return cues
    missing = [name for name in _CUE_PROTOTYPES if not getattr(cues, name)]
    if not missing:
        return cues
    sentences = iter_license_sentences(text)
    if not sentences:
        return cues
    embed = embed_fn or embed_texts
    prototypes: list[str] = []
    seen: set[str] = set()
    for name in missing:
        for sentence in _CUE_PROTOTYPES[name]:
            if sentence not in seen:
                seen.add(sentence)
                prototypes.append(sentence)
    try:
        vectors = embed(prototypes + sentences, model=model)
    except Exception:
        logger.exception("embed 线索补召回失败，保留正则线索")
        return cues
    if len(vectors) != len(prototypes) + len(sentences):
        logger.warning("embed 线索补召回返回长度不符，保留正则线索")
        return cues
    proto_vecs = {s: vectors[i] for i, s in enumerate(prototypes)}
    sent_vecs = vectors[len(prototypes) :]
    updates: dict[str, bool] = {}
    for name in missing:
        proto = _CUE_PROTOTYPES[name]
        query = _avg_vectors([proto_vecs[s] for s in proto])
        best_sim = -1.0
        best_sentence = ""
        for sentence, vec in zip(sentences, sent_vecs):
            sim = _cosine(query, vec)
            if sim > best_sim:
                best_sim = sim
                best_sentence = sentence
        if best_sim < min_sim or not _CUE_ANCHORS[name](best_sentence):
            continue
        updates[name] = True
        logger.info(
            "embed cue %s sim=%.3f tau=%.3f",
            name,
            best_sim,
            min_sim,
        )
    if not updates:
        return cues
    from dataclasses import replace

    return replace(cues, **updates)
