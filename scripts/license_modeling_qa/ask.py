"""调用 helper.call_llm，把 JSON 填进 LegalIR。"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from .extract import (
    NamedMention,
    _COPIES_SAME_LICENSE_RE,
    classify_named_licenses,
    amalgamate_follow_sentences,
    evidence_for_topics,
    is_short_fulltext,
    make_fulltext_evidence,
    with_amalgamate_follow_evidence,
    normalize_named_license,
)
from .gazetteer import (
    AUGMENT_TOPICS,
    COMBINE_TOPICS,
    COMPAT_TOPICS,
    DATA_REUSE_TOPICS,
    DISTILL_TOPICS,
    EMBED_TOPICS,
    MODIFY_TOPICS,
    NETWORK_TOPICS,
    OUTPUT_TOPICS,
    RESTRICT_TOPICS,
    STAT_TOPICS,
    TRAIN_TOPICS,
)
from .ir import (
    AmalgamateFacts,
    AugmentFacts,
    CombineFacts,
    CombineSituation,
    CompatFacts,
    EmbedFacts,
    GrantFacts,
    LegalIR,
    ModifyFacts,
    NetworkFacts,
    OutputFacts,
    RestrictFacts,
    StatFacts,
    TrainFacts,
)
from .questions import (
    SYSTEM,
    build_amalgamate_user,
    build_combine_user,
    build_compat_adopt_user,
    build_compat_user,
    build_embed_user,
    build_l0_user,
    build_modify_user,
    build_network_user,
    build_output_user,
    build_relicense_only_user,
    build_restrict_user,
    build_train_user,
)
from .types import EvidenceItem, PackDecision

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_L0_TOPICS = [
    "copyleft_sharealike",
    "modify_adapt",
    "notice_attribution",
    "source_disclose",
    "commercial",
    "sublicense",
    "patent",
    "grant",
    "redistribute",
    "trademark",
    "identity",
    "compat_convert",
]
_COMBINE_TOPIC_ORDER = [
    "combine_other_uses",
    "combine_agg",
    "combine_absorbed",
    "combine_database_assimilate",
    "exception_or_separate_work",
    "combine_link",
    "combine_mix",
    "adapt_share_conditions",
]
_GRANT_OPTS = {
    "use",
    "modify",
    "merge",
    "redistribute",
    "sublicense",
    "commercial_use",
    "patent_use",
}
_RESERVED_OPTS = {
    "trademark_use",
    "copyright",
    "patent",
    "trademark",
    "sublicense",
    "commercial_use",
    "patent_use",
    "redistribute",
}
_NOTICE_OPTS = {"notice", "license_text", "state_changes"}
_SITUATIONS = {"agg", "other_uses", "absorbed", "mixed_type", "db_assimilate"}
_FORMS = {"raw", "binary", "saas"}
_QA_THINKING_OFF = {"thinking": {"type": "disabled"}}
_QA_THINKING_ON = {"thinking": {"type": "enabled"}}
_QA_MAX_TOKENS = 4000
_QA_RETRY_MAX_TOKENS = 8000
_QA_LLM_TIMEOUT = 180
_QA_QUESTION_WORKERS = 8
_CALL_LOG_LOCK = threading.Lock()


def question_workers() -> int:
    """一份许可内部同时发出的问卷数。QA_QUESTION_WORKERS=1 时保持串行。"""
    raw = os.getenv("QA_QUESTION_WORKERS")
    if raw is None or not str(raw).strip():
        return _QA_QUESTION_WORKERS
    try:
        return max(1, int(raw))
    except ValueError:
        return _QA_QUESTION_WORKERS


def run_question_calls(jobs: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
    """没有先后依赖的问卷一起发。返回值按名字取，不按完成顺序。"""
    if not jobs:
        return {}
    if len(jobs) == 1 or question_workers() == 1:
        return {name: fn() for name, fn in jobs}
    workers = min(question_workers(), len(jobs))
    done: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fn): name for name, fn in jobs}
        for fut in as_completed(future_map):
            done[future_map[fut]] = fut.result()
    return done


def fill_ir(
    helper: Any,
    license_name: str,
    ir: LegalIR,
    evidence: list[EvidenceItem],
    packs: PackDecision,
    license_text: str = "",
) -> LegalIR:
    ir.evidence = evidence
    fulltext_mode = is_short_fulltext(license_text)
    full_ev = [make_fulltext_evidence(license_text)] if fulltext_mode else None
    if fulltext_mode:
        logging.info(
            "qa reading_mode=fulltext license=%s chars=%d",
            license_name,
            len(license_text or ""),
        )

    def _ask_ev(topics, *, limit: int = 15) -> list[EvidenceItem]:
        if full_ev is not None:
            return list(full_ev)
        return evidence_for_topics(evidence, topics, limit=limit)

    cite = evidence if not fulltext_mode else full_ev
    jobs: list[tuple[str, Callable[[], Any]]] = []

    def _enqueue(name: str, label: str, user: str) -> None:
        def run() -> Optional[dict]:
            return _ask_batch(helper, license_name, label, user)

        jobs.append((name, run))

    _enqueue(
        "l0",
        "qa_l0",
        build_l0_user(
            license_name,
            _ask_ev(_L0_TOPICS, limit=18),
            fulltext_mode=fulltext_mode,
        ),
    )
    if "P_combine" in packs.packs:
        _enqueue(
            "combine",
            "qa_combine",
            build_combine_user(
                license_name,
                _ask_ev(_COMBINE_TOPIC_ORDER, limit=15),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_modify" in packs.packs:
        _enqueue(
            "modify",
            "qa_modify",
            build_modify_user(
                license_name,
                _ask_ev(MODIFY_TOPICS),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_output" in packs.packs:
        _enqueue(
            "output",
            "qa_output",
            build_output_user(
                license_name,
                _ask_ev(list(OUTPUT_TOPICS | DISTILL_TOPICS), limit=10),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_compat" in packs.packs:
        _enqueue(
            "compat",
            "qa_compat",
            build_compat_user(
                license_name,
                _ask_ev(list(COMPAT_TOPICS), limit=8),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_restrict" in packs.packs:
        _enqueue(
            "restrict",
            "qa_restrict",
            build_restrict_user(
                license_name,
                _ask_ev(list(RESTRICT_TOPICS), limit=8),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_network" in packs.packs:
        _enqueue(
            "network",
            "qa_network",
            build_network_user(
                license_name,
                _ask_ev(list(NETWORK_TOPICS), limit=8),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_train" in packs.packs:
        _enqueue(
            "train",
            "qa_train",
            build_train_user(
                license_name,
                _ask_ev(list(TRAIN_TOPICS | DISTILL_TOPICS), limit=10),
                fulltext_mode=fulltext_mode,
            ),
        )
    if "P_embed" in packs.packs:
        _enqueue(
            "embed",
            "qa_embed",
            build_embed_user(
                license_name,
                _ask_ev(list(DATA_REUSE_TOPICS), limit=12),
                fulltext_mode=fulltext_mode,
            ),
        )
    saw_follow = False
    if "P_modify" in packs.packs or "P_combine" in packs.packs:
        amalg_ev = _ask_ev(
            set(MODIFY_TOPICS)
            | {"combine_absorbed", "combine_database_assimilate"},
            limit=12,
        )
        if fulltext_mode:
            saw_follow = bool(amalgamate_follow_sentences(license_text))
        else:
            amalg_ev, saw_follow = with_amalgamate_follow_evidence(
                amalg_ev, license_text
            )
        _enqueue(
            "amalgamate",
            "qa_amalgamate",
            build_amalgamate_user(
                license_name,
                amalg_ev,
                fulltext_mode=fulltext_mode,
            ),
        )
    done = run_question_calls(jobs)
    l0 = done.get("l0")
    if l0:
        ir.grant = _parse_grant(l0, cite)
        ir.l0_ok = True
    raw = done.get("combine")
    if raw:
        ir.combine = _parse_combine(raw, cite)
        ir.combine_ok = True
    raw = done.get("modify")
    if raw:
        ir.modify = _parse_modify(raw, cite)
        ir.modify_ok = True
    raw = done.get("output")
    if raw:
        ir.output = _parse_output(raw, cite)
        ir.output_ok = True
    raw = done.get("compat")
    if raw:
        extra = _parse_compat(raw, license_name)
        blocked = {item.lower() for item in ir.named.rejected or []}
        extra.allow = [sid for sid in extra.allow if sid.lower() not in blocked]
        ir.named.allow = _merge_ids(ir.named.allow, extra.allow)
        ir.named.ban = _merge_ids(ir.named.ban, extra.ban)
        ir.compat_ok = True
    raw = done.get("restrict")
    if raw:
        ir.restrict = _parse_restrict(raw, cite)
        ir.restrict_ok = True
    raw = done.get("network")
    if raw:
        ir.network = _parse_network(raw, cite)
        ir.network_ok = True
    raw = done.get("train")
    if raw:
        ir.train = _parse_train(raw, cite)
        ir.train_ok = True
    raw = done.get("embed")
    if raw:
        ir.embed = _parse_embed(raw, cite)
        ir.augment = _parse_augment(raw, cite)
        ir.stat = _parse_stat(raw, cite)
        ir.embed_ok = True
        ir.augment_ok = True
        ir.stat_ok = True
    raw = done.get("amalgamate")
    if raw:
        ir.amalgamate = _parse_amalgamate(raw)
        ir.amalgamate.saw_follow_sentence = saw_follow
        ir.amalgamate_ok = True
    _apply_relicense_questions(
        helper, license_name, ir, evidence, license_text, fulltext_mode
    )
    _correct_work_signals(ir)
    _append_block_log(license_name, _block_sources(ir))
    return ir


def _norm_relicense_stated(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, "", "n/a", "not_mentioned"):
        return None
    if value in ("yes", "true", True):
        return "yes"
    if value in ("no", "false", False):
        return "no"
    if value == "conditional":
        return "conditional"
    return None


def _apply_relicense_questions(
    helper: Any,
    license_name: str,
    ir: LegalIR,
    evidence: list[EvidenceItem],
    license_text: str,
    fulltext_mode: bool,
) -> None:
    """装配前改写修改和训练的换证。核对没有结论时保留原问卷答案。"""
    from .relicense_settle import relicense_evidence_packets, settle_relicense

    modify_real = (
        ir.modify_ok
        and ir.modify.bound in {"bound", "not_bound"}
    )
    train_real = (
        ir.train_ok
        and ir.train.mentioned
        and ir.train.bound in {"bound", "not_bound"}
    )
    amalgamate_real = (
        ir.amalgamate_ok
        and ir.amalgamate is not None
        and ir.amalgamate.mentioned
        and ir.amalgamate.bound in {"bound", "not_bound"}
    )
    if not modify_real and not train_real and not amalgamate_real:
        return
    modify_ev, train_ev = relicense_evidence_packets(license_text, evidence)
    jobs: list[tuple[str, Callable[[], Any]]] = []

    def _enqueue(kind: str, packet: list[EvidenceItem]) -> None:
        def run() -> tuple[Optional[str], Optional[str]]:
            payload = _ask_batch(
                helper,
                license_name,
                f"{kind}_relicense_only",
                build_relicense_only_user(
                    license_name, packet, kind=kind, fulltext_mode=fulltext_mode
                ),
            )
            stated = _norm_relicense_stated(
                (payload or {}).get("relicense") if isinstance(payload, dict) else None
            )
            settled = settle_relicense([item.quote or "" for item in packet], stated)
            return stated, settled

        jobs.append((kind, run))

    if modify_real:
        _enqueue("modify", modify_ev)
    if train_real:
        _enqueue("train", train_ev)
    if amalgamate_real:
        _enqueue("amalgamate", modify_ev or evidence)
    settled_by_kind = run_question_calls(jobs)
    if modify_real:
        stated, settled = settled_by_kind["modify"]
        if settled is not None:
            logging.info(
                "qa relicense_only license=%s kind=modify stated=%s settled=%s",
                license_name,
                stated,
                settled,
            )
            ir.modify.relicense = settled
    if train_real:
        stated, settled = settled_by_kind["train"]
        if settled is not None:
            logging.info(
                "qa relicense_only license=%s kind=train stated=%s settled=%s",
                license_name,
                stated,
                settled,
            )
            ir.train.relicense = settled
    if amalgamate_real:
        stated, settled = settled_by_kind["amalgamate"]
        ir.amalgamate.relicense_settled = settled is not None
        if settled is not None:
            logging.info(
                "qa relicense_only license=%s kind=amalgamate stated=%s settled=%s",
                license_name,
                stated,
                settled,
            )
            ir.amalgamate.relicense = settled


def _correct_work_signals(ir: LegalIR) -> None:
    """按定义句补上作品类型。线索没抽到时，不删问卷已经选出的类型。"""
    from .ir import resolved_work_types, store_work_signals

    before = ir.grant.work_signals
    ir.grant.work_signals = store_work_signals(
        resolved_work_types(ir.grant.work_signals, ir.cues)
    )
    if ir.grant.work_signals != before:
        overwrote = list(getattr(ir.grant, "rule_overwrote", []) or [])
        overwrote.append("work_signals")
        ir.grant.rule_overwrote = overwrote


def judge_compat_adoption(
    helper: Any,
    license_name: str,
    mentions: list[NamedMention],
) -> list[str]:
    """问模型：新作品整体能否采用候选许可。只接受候选列表里的 yes。"""
    if not mentions:
        return []
    sent = []
    seen: set[str] = set()
    for item in mentions:
        key = item.sid.lower()
        if key in seen:
            continue
        seen.add(key)
        sent.append((item.sid, item.quote))
    raw = _ask_batch(
        helper,
        license_name,
        "qa_compat_adopt",
        build_compat_adopt_user(license_name, sent),
    )
    if not raw:
        return []
    yes: list[str] = []
    allowed = {sid.lower(): sid for sid, _quote in sent}
    for item in _as_list(raw.get("items")):
        if not isinstance(item, dict):
            continue
        if str(item.get("adopt") or "").strip().lower() != "yes":
            continue
        sid = normalize_named_license(str(item.get("candidate") or ""))
        if sid and sid.lower() in allowed and sid not in yes:
            yes.append(allowed[sid.lower()])
    return yes


def resolve_named_licenses(helper: Any, license_name: str, text: str) -> CompatFacts:
    """规则锁定允许/拒绝；未决候选才按模型的 adopt=yes 进入 compat。"""
    found = classify_named_licenses(text, license_name)
    blocked = {sid.lower() for sid in found.rejected}
    model_yes = judge_compat_adoption(helper, license_name, found.mentions)
    llm_consulted = bool(found.mentions)
    allow: list[str] = []
    undecided = {item.sid.lower() for item in found.mentions if item.rule == "undecided"}
    for sid in found.allow:
        if sid.lower() in blocked or sid in allow:
            continue
        allow.append(sid)
    kept_yes: list[str] = []
    for sid in model_yes:
        if sid.lower() in blocked or sid.lower() not in undecided or sid in allow:
            continue
        allow.append(sid)
        kept_yes.append(sid)
    adopted: list[str] = []
    for item in found.mentions:
        if item.rule == "allow" and item.sid in allow and item.sid not in adopted:
            adopted.append(item.sid)
    for sid in kept_yes:
        if sid not in adopted:
            adopted.append(sid)
    named = CompatFacts(
        allow=allow,
        ban=list(found.ban),
        rejected=list(found.rejected),
        adopted=adopted,
    )
    named.llm_consulted = llm_consulted
    return named


def _qa_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _qa_temperature() -> float:
    raw = os.getenv("QA_TEMPERATURE")
    if raw is None or not str(raw).strip():
        return 0.1
    try:
        value = float(raw)
    except ValueError:
        return 0.1
    return value if value >= 0 else 0.1


def _qa_thinking_extra() -> dict[str, Any]:
    mode = (os.getenv("QA_THINKING") or "off").strip().lower()
    if mode in {"omit", "none", "na"}:
        return {}
    if mode in {"on", "enabled", "enable"}:
        return dict(_QA_THINKING_ON)
    return dict(_QA_THINKING_OFF)


def _usage_numbers(meta: dict[str, Any]) -> dict[str, Any]:
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if total is None and isinstance(prompt, int) and isinstance(completion, int):
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "usage": usage or None,
    }


def _block_sources(ir: Any) -> list[dict[str, Any]]:
    """进入装配的事实块：词表、问卷，或问卷之后被规则改写。"""
    rows: list[dict[str, Any]] = [
        {"block": "identity", "source": "lexicon", "rule_overwrote": False},
        {"block": "cues", "source": "lexicon", "rule_overwrote": False},
    ]
    named = ir.named
    if named and (named.allow or named.ban or named.rejected or getattr(named, "llm_consulted", False)):
        rows.append({
            "block": "named",
            "source": "questionnaire" if getattr(named, "llm_consulted", False) else "lexicon",
            "rule_overwrote": False,
        })
    if ir.l0_ok:
        fields = list(getattr(ir.grant, "rule_overwrote", []) or [])
        rows.append({
            "block": "grant",
            "source": "questionnaire",
            "rule_overwrote": bool(fields),
            "fields": fields,
        })
    for block, ok in (
        ("combine", ir.combine_ok),
        ("modify", ir.modify_ok),
        ("amalgamate", ir.amalgamate_ok),
        ("output", ir.output_ok),
        ("restrict", ir.restrict_ok),
        ("network", ir.network_ok),
        ("train", ir.train_ok),
        ("embed", ir.embed_ok),
        ("augment", ir.augment_ok),
        ("stat", ir.stat_ok),
    ):
        if ok:
            rows.append({"block": block, "source": "questionnaire", "rule_overwrote": False})
    return rows


def _append_block_log(license_name: str, blocks: list[dict[str, Any]]) -> None:
    path = (os.getenv("QA_BLOCK_LOG") or "").strip()
    if not path:
        return
    entered = len(blocks)
    questionnaire = [row for row in blocks if row["source"] == "questionnaire" and not row["rule_overwrote"]]
    record = {
        "record": "blocks",
        "license_name": license_name,
        "blocks": blocks,
        "entered": entered,
        "llm_blocks": len(questionnaire),
        "llm_rate": round(len(questionnaire) / entered, 4) if entered else None,
    }
    try:
        with _CALL_LOG_LOCK:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.debug("写入事实块来源失败", exc_info=True)


def _append_call_log(record: dict[str, Any]) -> None:
    path = (os.getenv("QA_CALL_LOG") or "").strip()
    if not path:
        return
    try:
        with _CALL_LOG_LOCK:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.debug("写入问卷调用记录失败", exc_info=True)


def _ask_batch(helper: Any, license_name: str, label: str, user: str) -> Optional[dict]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    max_tokens = _qa_int("QA_MAX_TOKENS", _QA_MAX_TOKENS)
    retry_max_tokens = _qa_int("QA_RETRY_MAX_TOKENS", _QA_RETRY_MAX_TOKENS)
    timeout = _qa_int("QA_LLM_TIMEOUT", _QA_LLM_TIMEOUT)
    thinking = _qa_thinking_extra()
    temperature = _qa_temperature()
    for attempt in range(2):
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "extra": thinking,
        }
        started = time.perf_counter()
        try:
            text = helper.call_llm(**kwargs)
        except TypeError:
            kwargs.pop("extra", None)
            kwargs.pop("timeout", None)
            text = helper.call_llm(**kwargs)
        elapsed_s = round(time.perf_counter() - started, 3)
        if hasattr(helper, "current_llm_meta"):
            meta = helper.current_llm_meta() or {}
        else:
            meta = getattr(helper, "last_llm_meta", None) or {}
        parsed = parse_json_object(text) if text else None
        _append_call_log(
            {
                "license_name": license_name,
                "label": label,
                "attempt": attempt,
                "model": getattr(helper, "model", None),
                "thinking": thinking.get("thinking"),
                "max_tokens": max_tokens,
                "timeout_s": timeout,
                "elapsed_s": elapsed_s,
                "parsed_ok": parsed is not None,
                "finish_reason": meta.get("finish_reason"),
                "content_chars": meta.get("content_chars", len(text or "")),
                "reasoning_chars": meta.get("reasoning_chars"),
                "http_error": bool(meta.get("http_error")),
                "exception": meta.get("exception"),
                **_usage_numbers(meta if isinstance(meta, dict) else {}),
            }
        )
        dump = text if text else json.dumps({"empty_content": True, **meta}, ensure_ascii=False)
        if hasattr(helper, "save_llm_output"):
            try:
                helper.save_llm_output(license_name, dump, f"{label}_attempt{attempt}")
            except Exception:
                logging.debug("保存问卷输出失败", exc_info=True)
        if parsed is not None:
            return parsed
        finish = meta.get("finish_reason")
        if not text:
            logging.warning(
                "qa 问卷 %s 第 %d 次空 content finish=%s reasoning_chars=%s",
                label,
                attempt + 1,
                finish,
                meta.get("reasoning_chars"),
            )
            if finish == "length":
                max_tokens = retry_max_tokens
        else:
            logging.warning(
                "qa 问卷 %s 第 %d 次 JSON 无法解析 finish=%s",
                label,
                attempt + 1,
                finish,
            )
    return None


def parse_json_object(text: str) -> Optional[dict]:
    if not text or not str(text).strip():
        return None
    raw = _JSON_FENCE.sub("", str(text).strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _unwrap(payload: dict, key: str) -> Any:
    node = payload.get(key)
    if isinstance(node, dict) and "value" in node:
        return node.get("value"), list(node.get("evidence_ids") or [])
    return node, []


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if value in ("", "none", "not_mentioned", "n/a"):
        return []
    return [value]


def _parse_work_signals(value: Any) -> str | list[str]:
    """作品类型可多选。单个合法值仍存成字符串，以兼容旧答案。"""
    allowed = {"software", "data", "model"}
    items = _as_list(value)
    out: list[str] = []
    for item in items:
        tok = str(item).strip().lower()
        if tok in allowed and tok not in out:
            out.append(tok)
    if not out:
        return "not_mentioned"
    if len(out) == 1:
        return out[0]
    return out


def _pick(value: Any, allowed: set[str], default: str = "not_mentioned") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in allowed:
        return text
    return default


def _parse_grant(payload: dict, evidence: Optional[list[EvidenceItem]] = None) -> GrantFacts:
    public, _ = _unwrap(payload, "L0_public")
    signals, _ = _unwrap(payload, "L0_work_signals")
    grants, _ = _unwrap(payload, "L0_grants")
    reserved, _ = _unwrap(payload, "L0_reserved")
    disclose, _ = _unwrap(payload, "L0_disclose")
    same, _ = _unwrap(payload, "L0_same_license_this")
    compat, _ = _unwrap(payload, "L0_compat_names")
    notice, _ = _unwrap(payload, "L0_notice")
    same_stated = _pick(same, {"yes", "no", "not_mentioned"})
    same_val = _resolve_same_license(same_stated, evidence or [])
    notice_stated = [n for n in _as_list(notice) if n in _NOTICE_OPTS]
    notice_list = _resolve_l0_notice(list(notice_stated), evidence or [])
    disclose_stated = _pick(disclose, {"yes", "no", "not_mentioned"})
    disclose_val = _resolve_disclose(disclose_stated, evidence or [])
    overwrote: list[str] = []
    if disclose_val != disclose_stated:
        overwrote.append("disclose")
    if notice_list != notice_stated:
        overwrote.append("notice")
    if same_val != same_stated:
        overwrote.append("same_license")
    facts = GrantFacts(
        public=_pick(public, {"yes", "no", "not_mentioned"}),
        work_signals=_parse_work_signals(signals),
        grants=[g for g in _as_list(grants) if g in _GRANT_OPTS],
        reserved=[r for r in _as_list(reserved) if r in _RESERVED_OPTS],
        disclose=disclose_val,
        same_license_this=same_val,
        compat_names=[str(x) for x in _as_list(compat) if str(x).strip()],
        notice=notice_list,
    )
    facts.rule_overwrote = overwrote
    return facts


_NOTICE_RETAIN_HINT = re.compile(
    r"copyright notice|attribution|"
    r"notice that refers to this(?:\s+public)?\s+licen[cs]e|"
    r"(?:retain|keep|preserve)\b.{0,200}notices?",
    re.IGNORECASE | re.DOTALL,
)
# 附证/标明本许可：要「copy of this License」等分发义务句。
# 不要用裸的「text of this License」——MG0/AFL/OSL 的
# 「You may modify the text of this License…」是改许可证文本身，不是附证义务。
_LICENSE_INDICATE_HINT = re.compile(
    r"licensed under this(?:\s+public)?\s+licen[cs]e|"
    r"(?:uri|url|hyperlink)\s+to,?\s+this(?:\s+public)?\s+licen[cs]e|"
    r"copy of this(?:\s+public)?\s+licen[cs]e|"
    r"include the text of this(?:\s+public)?\s+licen[cs]e|"
    r"this list of conditions|"
    r"reproduce the above copyright notice, this list of conditions|"
    r"include a complete copy of this licen[cs]e",
    re.IGNORECASE,
)
_LICENSE_DOC_FORK_HINT = re.compile(
    r"modify the text of this(?:\s+public)?\s+licen[cs]e|"
    r"Modification of This License",
    re.IGNORECASE,
)
_STATE_CHANGES_HINT = re.compile(
    r"indicate if you modified|previous modifications|"
    r"modified files to carry|stating that you (?:modified|changed)|"
    r"document how it differs|"
    r"describ(?:e|es|ing) how\b.{0,40}(?:differs|changes)",
    re.IGNORECASE,
)
_DO_NOT_ALTER_NOTICES_RE = re.compile(
    r"(?:may |shall |must )?not (?:remove|alter|change)\b.{0,80}"
    r"(?:notices|licen[cs]e)|"
    r"(?:notices|licen[cs]e text)\b.{0,40}(?:may |shall )?not be "
    r"(?:removed|altered|changed)",
    re.IGNORECASE | re.DOTALL,
)

# 二进制/目标码分发须提供源码（disclose），非仅「source and binary forms」并列条件。
_DISCLOSE_YES_RE = re.compile(
    r"Corresponding Source|"
    r"Executable Form.{0,200}Source Code Form|"
    r"Source Code Form.{0,120}Executable Form|"
    r"must also be made available as Source Code|"
    r"provide recipients? the source code|"
    r"how they can obtain a copy of such Source Code|"
    r"accompany the Program with a statement that the Source Code|"
    r"(?:object code|binary(?:\s+form)?|Executable Form).{0,160}"
    r"(?:must|shall)\b.{0,100}"
    r"(?:(?:also )?be made available|provide|accompany|offer|inform).{0,100}"
    r"(?:source code|Source Code(?: Form)?|Corresponding Source)",
    re.IGNORECASE | re.DOTALL,
)
_DISCLOSE_FALSE_FRIEND_RE = re.compile(
    r"Redistribution and use in source and binary forms|"
    r"Redistributions of source code must retain|"
    r"Redistributions in binary form must reproduce|"
    r"in source and binary forms, with or without",
    re.IGNORECASE,
)


def _blob_has_disclose_obligation(blob: str) -> bool:
    if not (blob or "").strip():
        return False
    if not _DISCLOSE_YES_RE.search(blob):
        return False
    # 仅 BSD 式并列条件、无强义务句 → 不抬
    strong = re.search(
        r"Corresponding Source|provide recipients? the source code|"
        r"must also be made available as Source Code|"
        r"how they can obtain a copy of such Source Code|"
        r"(?:must|shall)\b.{0,100}(?:source code|Source Code|Corresponding Source)",
        blob,
        re.IGNORECASE | re.DOTALL,
    )
    if _DISCLOSE_FALSE_FRIEND_RE.search(blob) and not strong:
        return False
    return bool(
        re.search(
            r"distribut|convey|Executable Form|object code|binary|made available|"
            r"provide recipients|inform recipients|accompany",
            blob,
            re.IGNORECASE,
        )
    )


def _resolve_disclose(stated: str, evidence: list[EvidenceItem]) -> str:
    """有「目标码/二进制分发须提供源码」结构句时抬 disclose=yes。"""
    blob = " ".join((e.quote or "") for e in evidence)
    if _blob_has_disclose_obligation(blob):
        return "yes"
    # BSD 式 source/binary 并列条件被误答 yes → 压回
    if stated == "yes" and _DISCLOSE_FALSE_FRIEND_RE.search(blob):
        if not re.search(
            r"Corresponding Source|provide recipients? the source code|"
            r"must also be made available as Source Code",
            blob,
            re.IGNORECASE,
        ):
            return "not_mentioned"
    return stated


def _blob_indicates_license_text(blob: str) -> bool:
    """分发须附/标明本许可；跳过「允许修改本许可证文本」窗口内的命中。"""
    text = blob or ""
    for match in _LICENSE_INDICATE_HINT.finditer(text):
        window = text[max(0, match.start() - 120) : match.end() + 40]
        if _LICENSE_DOC_FORK_HINT.search(window):
            continue
        return True
    if _NOTICE_RETAIN_HINT.search(text) and re.search(
        r"this list of conditions", text, re.IGNORECASE
    ):
        return True
    return False


def _resolve_l0_notice(stated: list[str], evidence: list[EvidenceItem]) -> list[str]:
    """Attribution 条件簇：引文同时有保留声明与标明许可时补全 notice + license_text。"""
    out = list(stated)
    blob = " ".join((e.quote or "") for e in evidence)
    if _NOTICE_RETAIN_HINT.search(blob) and "notice" not in out:
        out.append("notice")
    if _blob_indicates_license_text(blob) and "license_text" not in out:
        out.append("license_text")
    if _STATE_CHANGES_HINT.search(blob) and "state_changes" not in out:
        out.append("state_changes")
    if (
        "state_changes" in out
        and _DO_NOT_ALTER_NOTICES_RE.search(blob)
        and not _STATE_CHANGES_HINT.search(blob)
    ):
        out = [item for item in out if item != "state_changes"]
    if _DO_NOT_ALTER_NOTICES_RE.search(blob) and "notice" not in out:
        out.append("notice")
    return [n for n in out if n in _NOTICE_OPTS]


_YES_SAME_LICENSE = (
    re.compile(r"share\s*alike|sharealike", re.IGNORECASE),
    re.compile(r"\bsame terms\b", re.IGNORECASE),
    re.compile(r"license the entire work", re.IGNORECASE),
    re.compile(r"must be licensed under this", re.IGNORECASE),
    re.compile(r"must be made available under this (?:licen[cs]e|agreement)", re.IGNORECASE),
    re.compile(r"whole of the work to be licensed", re.IGNORECASE),
    re.compile(
        r"(?:modif(?:y|ied|ications|ied versions?)|work based on(?: the (?:library|work|program))?|"
        r"derivative works?)"
        r".{0,240}under the terms of (?:this (?:licen[cs]e|public licen[cs]e)|section \d+)",
        re.IGNORECASE | re.DOTALL,
    ),
    # 源码形式仅可按本许可再分发（递延/文件级 copyleft 常见）
    re.compile(
        r"(?:source code form|in source code form).{0,120}only under this licen[cs]e|"
        r"only under this licen[cs]e by including a complete copy|"
        r"may do so only under this licen[cs]e",
        re.IGNORECASE | re.DOTALL,
    ),
    # 字体/素材：须完全在本许可下分发，不得换用其他许可
    re.compile(
        r"must be distributed entirely under this licen[cs]e|"
        r"must not be distributed under any other licen[cs]e|"
        r"cannot be released under any other type of licen[cs]e|"
        r"remain under this licen[cs]e",
        re.IGNORECASE,
    ),
    _COPIES_SAME_LICENSE_RE,
)
_NO_DIFFERENT_TERMS = re.compile(
    r"additional or different licen[cs]e terms|released under different licensing terms",
    re.IGNORECASE,
)
_NO_ADAPTER = re.compile(r"adapter'?s licen[cs]e", re.IGNORECASE)
_NOTICE_ONLY = re.compile(
    r"(?:give|provide|include|attach|distribute|supply)\b.{0,80}a copy of this licen[cs]e|"
    r"a copy of this licen[cs]e",
    re.IGNORECASE | re.DOTALL,
)
# 仅通知/广告/命名条件，不是同许可再授权
# 保留条件清单是通知义务。禁背书、保留名称、广告声明不投同许可的反对票。
_NO_NOTICE_OR_ADVERTISING = re.compile(
    r"this list of conditions|"
    r"copyright notice.{0,80}disclaimer",
    re.IGNORECASE,
)


def _vote_same_license(quote: str) -> Optional[str]:
    """一句一票。yes 优先于 no；纯通知句投 no，但不能取消另一句的 yes。"""
    text = quote or ""
    yes_hit = any(pattern.search(text) for pattern in _YES_SAME_LICENSE)
    if yes_hit:
        # 同句仅有 notice/广告条件、无「licensed under this」类义务 → 不投 yes
        has_license_under = re.search(
            r"licensed under this|under the terms of this|only under this licen|"
            r"made available under this (?:licen|agreement)|"
            r"share\s*alike|same terms|entire work|"
            r"entirely under this licen|under any other licen|remain under this licen",
            text,
            re.IGNORECASE,
        )
        if _NO_NOTICE_OR_ADVERTISING.search(text) and not has_license_under:
            return "no"
        return "yes"
    if _NO_DIFFERENT_TERMS.search(text):
        return "no"
    if _NO_ADAPTER.search(text) and not re.search(
        r"share\s*alike|sharealike", text, re.IGNORECASE
    ):
        return "no"
    if _NOTICE_ONLY.search(text):
        return "no"
    if _NO_NOTICE_OR_ADVERTISING.search(text) and not re.search(
        r"licensed under this|under the terms of this|share\s*alike",
        text,
        re.IGNORECASE,
    ):
        return "no"
    return None


def _resolve_same_license(stated: str, evidence: list[EvidenceItem]) -> str:
    votes = [
        vote
        for vote in (_vote_same_license(e.quote or "") for e in evidence)
        if vote
    ]
    if "yes" in votes:
        return "yes"
    if "no" in votes:
        return "no"
    return stated


def _parse_combine(payload: dict, evidence: list[EvidenceItem]) -> CombineFacts:
    c0, c0_ids = _unwrap(payload, "C0")
    c0_yes = _pick(c0, {"yes", "no"}, default="no") == "yes"
    gazetteer_ids = [e.id for e in evidence if e.topic in COMBINE_TOPICS]
    if c0_yes and not c0_ids:
        c0_ids = gazetteer_ids[:8]
    mentioned = c0_yes and bool(c0_ids)
    c1, c1_ids = _unwrap(payload, "C1")
    sit_ids = [s for s in _as_list(c1) if s in _SITUATIONS]
    mix_ids = [e.id for e in evidence if e.topic == "combine_mix"]
    if mix_ids and "mixed_type" not in sit_ids:
        sit_ids.append("mixed_type")
    if mix_ids:
        mentioned = True
        if not c0_ids:
            c0_ids = mix_ids[:8]
    db_ids = [e.id for e in evidence if e.topic == "combine_database_assimilate"]
    if db_ids and "db_assimilate" not in sit_ids:
        sit_ids.append("db_assimilate")
    if db_ids:
        mentioned = True
        if not c0_ids:
            c0_ids = db_ids[:8]
    c2, c2_ids = _unwrap(payload, "C2")
    if not mentioned:
        return CombineFacts(mentioned=False)
    if "other_uses" in sit_ids:
        other = _pick(c2, {"yes", "no", "not_mentioned", "n/a"}, default="not_mentioned")
        if other == "yes" and not c2_ids and not gazetteer_ids:
            other = "not_mentioned"
        other = _resolve_c2(other, evidence, sit_ids)
    else:
        other = "n/a"
    cite_ids = list(c1_ids or c0_ids or gazetteer_ids[:8])
    situations: list[CombineSituation] = []
    for sid in sit_ids:
        forms = _forms_for(_combine_form_value(payload, sid))
        bound, relic, restr = _combine_slot(payload, sid)
        bound_val = _pick(bound, {"bound", "not_bound", "not_mentioned"})
        relic_val = _pick(
            relic,
            {"yes", "no", "conditional", "not_mentioned", "true", "false"},
        )
        if sid == "mixed_type":
            bound_val = _resolve_mixed_bound(bound_val, evidence)
        if sid == "db_assimilate":
            if bound_val in {"not_mentioned", "n/a", ""}:
                bound_val = "bound"
            relic_val = _resolve_db_assimilate_relicense(relic_val, evidence)
            if not forms:
                forms = ["raw", "binary"]
        situations.append(
            CombineSituation(
                id=sid,
                forms=forms,
                bound=bound_val,
                relicense=relic_val,
                same_license=other if sid == "other_uses" else "not_mentioned",
                restrictions=_as_list(restr),
                evidence_ids=cite_ids,
            )
        )
    return CombineFacts(
        mentioned=True,
        situations=situations,
        other_work_must_same_license=other,
    )


def _combine_form_value(payload: dict, sid: str) -> Any:
    val, _ = _unwrap(payload, f"C3_{sid}")
    if val is not None:
        return val
    nested = payload.get("C3")
    if isinstance(nested, dict) and "value" in nested and isinstance(nested.get("value"), dict):
        nested = nested["value"]
    if isinstance(nested, dict):
        return nested.get(sid)
    return None


def _combine_slot(payload: dict, sid: str) -> tuple[Any, Any, Any]:
    bound, _ = _unwrap(payload, f"C4_{sid}_bound")
    relic, _ = _unwrap(payload, f"C4_{sid}_relicense")
    restr, _ = _unwrap(payload, f"C4_{sid}_restrictions")
    if bound is not None or relic is not None or restr:
        return bound, relic, restr
    nested = payload.get("C4")
    if isinstance(nested, dict) and "value" in nested and isinstance(nested.get("value"), dict):
        nested = nested["value"]
    slot = nested.get(sid) if isinstance(nested, dict) else None
    if isinstance(slot, dict):
        return slot.get("bound"), slot.get("relicense"), slot.get("restrictions")
    return None, None, None


_OTHER_C2_TOPICS = frozenset({"combine_other_uses", "combine_link"})
_INFECT_C2_TOPICS = frozenset({"combine_other_uses", "combine_link", "combine_absorbed"})
_AUP_TOPICS = frozenset({"use_restriction_aup"})


def _quotes_for(
    evidence: list[EvidenceItem],
    topics: frozenset[str],
    *,
    sources: Optional[frozenset[str]] = None,
) -> str:
    parts: list[str] = []
    for e in evidence:
        if e.topic not in topics:
            continue
        if sources is not None and (e.source or "regex") not in sources:
            continue
        parts.append((e.quote or "").lower())
    return " ".join(parts)


def _has_strong_infect_quote(evidence: list[EvidenceItem]) -> bool:
    """真传染句：整作品须按本许可再许可。embed 引文不单独定案。"""
    infect = _quotes_for(
        evidence, _INFECT_C2_TOPICS, sources=frozenset({"regex", "fulltext"})
    )
    if "license the entire work" in infect:
        return True
    return "entire work" in infect and "under this" in infect


def _has_aup_evidence(evidence: list[EvidenceItem]) -> bool:
    return any(e.topic in _AUP_TOPICS for e in evidence)


def _combine_c2_evidence_embed_only(evidence: list[EvidenceItem]) -> bool:
    items = [
        e
        for e in evidence
        if e.topic in _INFECT_C2_TOPICS or e.topic in _OTHER_C2_TOPICS
    ]
    return bool(items) and all((e.source or "regex") == "embed" for e in items)


def _resolve_mixed_bound(stated: str, evidence: list[EvidenceItem]) -> str:
    blob = _quotes_for(evidence, frozenset({"combine_mix"}))
    if "does not bring" in blob or (
        "mere aggregation" in blob and "linguistic" in blob
    ):
        return "not_bound"
    return stated


def _resolve_db_assimilate_relicense(
    stated: str, evidence: list[EvidenceItem]
) -> str:
    """Adapter/Share 条件 → conditional；ShareAlike 族 → no；否则默认 conditional。"""
    if stated in {"yes", "no", "conditional", "true", "false"}:
        return stated
    blob = _quotes_for(
        evidence,
        frozenset({"adapt_share_conditions", "copyleft_sharealike"}),
    )
    if "sharealike" in blob or "must be a creative commons" in blob:
        return "no"
    if "adapter's license" in blob or "must not prevent recipients" in blob:
        return "conditional"
    return "conditional"


def _resolve_c2(stated: str, evidence: list[EvidenceItem], sit_ids: list[str]) -> str:
    """C2 以「使用/链接」引文为准；假传染（AUP / 仅 embed）压否，真传染句可覆盖模型。"""
    if "other_uses" not in sit_ids:
        return "n/a"
    other = _quotes_for(evidence, _OTHER_C2_TOPICS)
    if "terms of your choice" in other:
        return "no"
    if _has_strong_infect_quote(evidence):
        return "yes"
    if stated == "yes" and (
        _has_aup_evidence(evidence) or _combine_c2_evidence_embed_only(evidence)
    ):
        return "no"
    if stated in {"yes", "no"}:
        return stated
    return stated


def _forms_for(value: Any) -> list[str]:
    if value in (None, "not_mentioned", "n/a", ""):
        return []
    if value == "same_all":
        return ["raw", "binary"]
    items = _as_list(value)
    return [f for f in items if f in _FORMS]


def _parse_modify(payload: dict, evidence: Optional[list[EvidenceItem]] = None) -> ModifyFacts:
    bound, ids = _unwrap(payload, "T1_bound")
    relicense, _ = _unwrap(payload, "T1_relicense")
    same, _ = _unwrap(payload, "T1_same_license")
    restr, _ = _unwrap(payload, "T1_restrictions")
    forms, _ = _unwrap(payload, "T1_forms")
    form_list = [f for f in _as_list(forms) if f in _FORMS] or ["raw", "binary"]
    same_val = _pick(same, {"yes", "no", "not_mentioned"})
    same_val = _resolve_same_license(same_val, evidence or [])
    relic_val = _pick(
        relicense, {"yes", "no", "conditional", "not_mentioned", "true", "false"}
    )
    relic_val = _resolve_modify_relicense(relic_val, same_val, evidence or [])
    return ModifyFacts(
        bound=_pick(bound, {"bound", "not_bound", "not_mentioned"}),
        relicense=relic_val,
        same_license=same_val,
        restrictions=_as_list(restr),
        forms=form_list,
        evidence_ids=list(ids or []),
    )


def _parse_amalgamate(payload: dict) -> AmalgamateFacts:
    mentioned_raw, ids = _unwrap(payload, "F0")
    mentioned = _pick(mentioned_raw, {"yes", "no"}) == "yes"
    if not mentioned:
        return AmalgamateFacts(mentioned=False, evidence_ids=list(ids or []))
    bound, _ = _unwrap(payload, "F1_bound")
    relicense, _ = _unwrap(payload, "F2_relicense")
    restr, _ = _unwrap(payload, "F3_restrictions")
    same, _ = _unwrap(payload, "F4_same_as_modify")
    allowed = {"include_notice", "include_license", "state_changes"}
    return AmalgamateFacts(
        mentioned=True,
        bound=_pick(bound, {"bound", "not_bound", "not_mentioned"}),
        relicense=_pick(
            relicense, {"yes", "no", "conditional", "not_mentioned", "true", "false"}
        ),
        same_as_modify=_pick(same, {"yes", "no", "not_mentioned"}),
        restrictions=[item for item in _as_list(restr) if item in allowed],
        restrictions_stated="F3_restrictions" in payload,
        evidence_ids=list(ids or []),
    )


_SHARE_ALIKE_HINT = re.compile(
    r"share\s*alike|sharealike|compatible licen[cs]e|"
    r"must be a creative commons.{0,40}share",
    re.IGNORECASE,
)


_CONDITIONAL_RELICENSE_RE = re.compile(
    # 可换证但受限：自选且不矛盾 / 兼容集 / Secondary / 合规许可
    r"(?:any )?licen[cs]e of your choice that does not contradict|"
    r"does not contradict the terms|"
    r"compatible licen[cs]e|"
    r"Secondary Licen[cs]e|"
    r"under the terms of (?:this Licen[cs]e|Agreement) or (?:a |such )?Secondary|"
    r"under a licen[cs]e that complies with this licen[cs]e",
    re.IGNORECASE,
)
# 「须按本许可」本身是同许可义务，不是有条件换证
_MUST_THIS_LICENSE_RE = re.compile(
    r"under the terms of this (?:Licen[cs]e|Agreement)|"
    r"only under this licen[cs]e|"
    r"entirely under this licen[cs]e|"
    r"must not be distributed under any other licen[cs]e",
    re.IGNORECASE,
)


def _resolve_modify_relicense(
    stated: str, same_license: str, evidence: list[EvidenceItem]
) -> str:
    """Adapter's License（无 ShareAlike）→ relicense=no；兼容/不矛盾换证 → conditional。"""
    blob = " ".join((e.quote or "") for e in evidence)
    has_adapter = bool(_NO_ADAPTER.search(blob))
    has_sa = bool(_SHARE_ALIKE_HINT.search(blob))
    if has_adapter and not has_sa:
        return "no"
    if same_license == "no" and stated == "conditional" and has_adapter:
        return "no"
    # 可选任意许可但不得与本许可矛盾 / 兼容许可集 → conditional（非无条件 true）
    if _CONDITIONAL_RELICENSE_RE.search(blob) and not (has_adapter and not has_sa):
        return "conditional"
    # 强同许可套话：不得因「under the terms of this License」误抬/误留 conditional
    if same_license == "yes" and _MUST_THIS_LICENSE_RE.search(blob):
        if not _CONDITIONAL_RELICENSE_RE.search(blob):
            if stated in {"conditional", "yes", "true", "not_mentioned", ""}:
                return "no"
    return stated


def _parse_output(payload: dict, evidence: list[EvidenceItem]) -> OutputFacts:
    o0, o0_ids = _unwrap(payload, "O0")
    gazetteer_ids = [
        e.id for e in evidence if e.topic in OUTPUT_TOPICS or e.topic in DISTILL_TOPICS
    ]
    mentioned = _pick(o0, {"yes", "no"}, default="no") == "yes"
    if mentioned and not o0_ids:
        o0_ids = gazetteer_ids[:4]
    improve, _ = _unwrap(payload, "O3")
    improve_val = _pick(
        improve, {"yes", "no", "not_mentioned", "n/a"}, default="not_mentioned"
    )
    improve_val = _resolve_improve_other(improve_val, evidence)
    if improve_val == "n/a":
        improve_val = "not_mentioned"
    if not mentioned:
        return OutputFacts(
            mentioned=False,
            improve_other=improve_val,
            evidence_ids=list(o0_ids or gazetteer_ids[:4]),
        )
    bound, bound_ids = _unwrap(payload, "O1")
    rules, _ = _unwrap(payload, "O2")
    bound_val = _pick(
        bound,
        {"bound", "not_bound", "only_if_output_is_covered", "n/a", "not_mentioned"},
        default="not_mentioned",
    )
    bound_val = _resolve_output_bound(bound_val, evidence)
    use_val = _pick(rules, {"yes", "no", "n/a", "not_mentioned"})
    if _output_use_rules_forced(evidence):
        use_val = "yes"
    return OutputFacts(
        mentioned=True,
        bound=bound_val,
        use_rules=use_val,
        improve_other=improve_val,
        evidence_ids=list(bound_ids or o0_ids or gazetteer_ids[:4]),
    )


def _output_use_rules_forced(evidence: list[EvidenceItem]) -> bool:
    """产物使用须遵守许可/使用限制的正面句 → 强制 O2=yes。

    不因「claims no rights / solely responsible」单独抬升（无 contravene 等约束句时）。
    允许「No use of the Output can」与「contravene…」被切成相邻单元。
    """
    out_items = [e for e in evidence if e.topic in OUTPUT_TOPICS]
    if not out_items:
        return False
    out_blob = " ".join((e.quote or "").lower() for e in out_items)
    # 切段后 contravene 可能落在无 output 字样的单元，并入扫描窗
    stitch = " ".join(
        (e.quote or "").lower()
        for e in evidence
        if "contravene" in (e.quote or "").lower()
    )
    blob = f"{out_blob} {stitch}".strip()
    if "use-based" in out_blob and "output" in out_blob:
        return True
    if re.search(r"no use of the outputs?\s+can\b", out_blob):
        return True
    if "no use of the output can contravene" in blob:
        return True
    if "contravene" in blob and "output" in out_blob and (
        "no use of the output" in blob or "use of the output" in blob
    ):
        return True
    if re.search(
        r"use of the outputs?.{0,100}(?:contravene|subject to|comply with|in accordance with)",
        blob,
    ):
        return True
    return False


def _resolve_output_bound(stated: str, evidence: list[EvidenceItem]) -> str:
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in OUTPUT_TOPICS
    )
    if "constitutes a covered work" in blob or "only if the output" in blob:
        return "only_if_output_is_covered"
    if "not deemed model derivatives" in blob or "outputs by themselves are not" in blob:
        if stated not in {"bound", "only_if_output_is_covered"}:
            return "not_bound"
    # OpenRAIL 等：不主张 Output 权利 ⇒ 非整包 bound（不否定 O2 使用限制）
    if "claims no rights" in blob and "output" in blob:
        if stated != "only_if_output_is_covered":
            return "not_bound"
    if stated in {"bound", "not_bound", "only_if_output_is_covered"}:
        return stated
    return stated


def _resolve_improve_other(stated: str, evidence: list[EvidenceItem]) -> str:
    blob = " ".join(
        (e.quote or "").lower()
        for e in evidence
        if e.topic in OUTPUT_TOPICS or e.topic in DISTILL_TOPICS
    )
    if "improve any other" in blob and "model" in blob:
        return "no"
    if stated in {"yes", "no", "not_mentioned"}:
        return stated
    return stated


def _merge_ids(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            key = str(item).strip()
            if not key:
                continue
            low = key.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(key)
    return out


def _parse_compat(payload: dict, self_id: str) -> CompatFacts:
    k0, _ = _unwrap(payload, "K0")
    names, _ = _unwrap(payload, "K1")
    bans, _ = _unwrap(payload, "K2")
    mentioned = _pick(k0, {"yes", "no"}, default="no") == "yes"
    allow: list[str] = []
    if mentioned:
        for raw in _as_list(names):
            sid = normalize_named_license(str(raw))
            if sid and sid.lower() != (self_id or "").lower():
                allow.append(sid)
    ban: list[str] = []
    for raw in _as_list(bans):
        sid = normalize_named_license(str(raw))
        if sid and sid.lower() != (self_id or "").lower():
            ban.append(sid)
        stem = str(raw).strip().upper()
        if stem in {"GPL", "AGPL", "LGPL", "GFDL"} and stem.lower() != (self_id or "").split("-")[0].lower():
            ban.append(stem)
    return CompatFacts(allow=_merge_ids(allow), ban=_merge_ids(ban))


_PASS_ON_CUES = (
    "enforceable provision",
    "shall require all of",
    "include the use restriction",
    "must include the use restriction",
    "subject to paragraph 5",
    "subject to the use restriction",
    "inconsistent with the terms of this license",
    "inconsistent with the terms of this agreement",
    "offer or impose any terms",
    "pass on these use restrictions",
    "pass on the use restrictions",
)


def _parse_restrict(payload: dict, evidence: list[EvidenceItem]) -> RestrictFacts:
    r0, r0_ids = _unwrap(payload, "R0")
    gazetteer_ids = [e.id for e in evidence if e.topic in RESTRICT_TOPICS]
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in RESTRICT_TOPICS
    )
    mentioned = _pick(r0, {"yes", "no"}, default="no") == "yes" or bool(gazetteer_ids)
    if not mentioned and (
        "use-based restriction" in blob
        or "acceptable use" in blob
        or "prohibited uses" in blob
        or "restricted uses" in blob
    ):
        mentioned = True
    if mentioned and not r0_ids:
        r0_ids = gazetteer_ids[:4]
    if not mentioned:
        return RestrictFacts(mentioned=False)
    pass_on, _ = _unwrap(payload, "R1")
    runtime, _ = _unwrap(payload, "R2")
    pass_val = _pick(
        pass_on, {"yes", "no", "n/a", "not_mentioned"}, default="not_mentioned"
    )
    run_val = _pick(
        runtime, {"yes", "no", "n/a", "not_mentioned"}, default="not_mentioned"
    )
    if any(cue in blob for cue in _PASS_ON_CUES):
        pass_val = "yes"
    if "use-based" in blob and "running" in blob:
        run_val = "yes"
    if pass_val == "n/a":
        pass_val = "not_mentioned"
    if run_val == "n/a":
        run_val = "not_mentioned"
    return RestrictFacts(
        mentioned=True,
        pass_on=pass_val,
        runtime=run_val,
        evidence_ids=list(r0_ids or gazetteer_ids[:4]),
    )


def _parse_network(payload: dict, evidence: list[EvidenceItem]) -> NetworkFacts:
    n0, n0_ids = _unwrap(payload, "N0")
    gazetteer_ids = [e.id for e in evidence if e.topic in NETWORK_TOPICS]
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in NETWORK_TOPICS
    )
    all_blob = " ".join((e.quote or "").lower() for e in evidence)
    mentioned = _pick(n0, {"yes", "no"}, default="no") == "yes"
    if not mentioned and (
        "interacting with it remotely" in blob
        or "remote network interaction" in blob
        or "remotely through a computer network" in blob
        or "hosted service" in blob
        or "api-based" in blob
        or "remotely accessible" in blob
        or "over a network" in blob
        or "as a distribution" in blob
        or "external deployment" in blob
        or "application intended for use over a network" in blob
    ):
        mentioned = True
    # 仅「从网络服务器提供对应源码」≠ 远程交互 / saas 形态
    source_only = "network server at no charge" in all_blob or (
        "corresponding source" in all_blob and "network server" in all_blob
    )
    remote_cues = (
        "remotely" in all_blob
        or "hosted" in all_blob
        or "over a network" in all_blob
        or "as a distribution" in all_blob
        or "external deployment" in all_blob
        or "api-based" in all_blob
        or "as a service" in all_blob
    )
    if source_only and not remote_cues:
        mentioned = False
    if mentioned and not n0_ids:
        n0_ids = gazetteer_ids[:4]
    if not mentioned:
        return NetworkFacts(mentioned=False)
    n1, _ = _unwrap(payload, "N1")
    remote = _pick(
        n1,
        {"yes", "no", "only_if_modified", "n/a", "not_mentioned"},
        default="not_mentioned",
    )
    if "interacting with it remotely" in blob or "remotely through a computer network" in blob:
        if "if you modify" in blob:
            remote = "only_if_modified"
        elif remote not in {"yes", "only_if_modified"}:
            remote = "yes"
    # 远程提供视同分发 → 与 convey 相同义务
    if remote not in {"yes", "only_if_modified"} and (
        "as a distribution" in blob
        or ("over a network" in blob and "treat" in blob)
        or "external deployment" in blob
    ):
        remote = "yes"
    if remote == "n/a":
        remote = "not_mentioned"
    return NetworkFacts(
        mentioned=True,
        remote_same_as_convey=remote,
        evidence_ids=list(n0_ids or gazetteer_ids[:4]),
    )


_TRAIN_BOUND_CUES = (
    "transfer of patterns of the weights",
    "derivatives of the model",
    "model derivatives",
    "distillation methods",
)
_TRAIN_MENTION_CUES = (
    "fine-tun",
    "finetun",
    "transfer of patterns of the weights",
    "running, training",
    "include creating any content with",
)
_TRAIN_DEF_ONLY = (
    "trained model weights",
    "training-enabling",
)


def _parse_train(payload: dict, evidence: list[EvidenceItem]) -> TrainFacts:
    m0, m0_ids = _unwrap(payload, "M0")
    gazetteer_ids = [
        e.id for e in evidence if e.topic in TRAIN_TOPICS or e.topic in DISTILL_TOPICS
    ]
    blob = " ".join(
        (e.quote or "").lower()
        for e in evidence
        if e.topic in TRAIN_TOPICS or e.topic in DISTILL_TOPICS
    )
    mentioned = _pick(m0, {"yes", "no"}, default="no") == "yes"
    if any(cue in blob for cue in _TRAIN_MENTION_CUES):
        mentioned = True
    if mentioned and any(cue in blob for cue in _TRAIN_DEF_ONLY):
        if not any(cue in blob for cue in _TRAIN_MENTION_CUES) and not any(
            cue in blob for cue in _TRAIN_BOUND_CUES
        ):
            mentioned = False
    if mentioned and not m0_ids:
        m0_ids = gazetteer_ids[:4]
    if not mentioned:
        return TrainFacts(mentioned=False)
    bound, _ = _unwrap(payload, "M1")
    relic, _ = _unwrap(payload, "M2")
    bound_val = _pick(bound, {"bound", "not_bound", "n/a", "not_mentioned"})
    if any(cue in blob for cue in _TRAIN_BOUND_CUES):
        bound_val = "bound"
    if bound_val == "n/a":
        bound_val = "not_mentioned"
    relic_val = _pick(
        relic, {"yes", "no", "conditional", "n/a", "not_mentioned", "true", "false"}
    )
    if relic_val == "n/a":
        relic_val = "not_mentioned"
    return TrainFacts(
        mentioned=True,
        bound=bound_val,
        relicense=relic_val,
        evidence_ids=list(m0_ids or gazetteer_ids[:4]),
    )


def _parse_embed(payload: dict, evidence: list[EvidenceItem]) -> EmbedFacts:
    e0, e0_ids = _unwrap(payload, "E0")
    gazetteer_ids = [e.id for e in evidence if e.topic in EMBED_TOPICS]
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in EMBED_TOPICS
    )
    mentioned = _pick(e0, {"yes", "no"}, default="no") == "yes"
    if "adapted material" in blob or "licensed material is translated" in blob:
        mentioned = True
    if mentioned and not e0_ids:
        e0_ids = gazetteer_ids[:4]
    if not mentioned:
        return EmbedFacts(mentioned=False)
    bound, _ = _unwrap(payload, "E1")
    relic, _ = _unwrap(payload, "E2")
    bound_val = _pick(
        bound, {"bound", "not_bound", "translation", "n/a", "not_mentioned"}
    )
    kind = "not_mentioned"
    if bound_val == "translation" or "translated" in blob:
        kind = "translation"
        if bound_val not in {"bound", "not_bound"}:
            bound_val = "translation"
    if bound_val == "n/a":
        bound_val = "not_mentioned"
    relic_val = _pick(
        relic, {"yes", "no", "conditional", "n/a", "not_mentioned", "true", "false"}
    )
    if relic_val == "n/a":
        relic_val = "not_mentioned"
    return EmbedFacts(
        mentioned=True,
        bound=bound_val,
        result_kind=kind,
        relicense=relic_val,
        evidence_ids=list(e0_ids or gazetteer_ids[:4]),
    )


def _parse_augment(payload: dict, evidence: list[EvidenceItem]) -> AugmentFacts:
    a0, a0_ids = _unwrap(payload, "A0")
    gazetteer_ids = [e.id for e in evidence if e.topic in AUGMENT_TOPICS]
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in AUGMENT_TOPICS
    )
    mentioned = _pick(a0, {"yes", "no"}, default="no") == "yes"
    if any(cue in blob for cue in ("remix", "altered, arranged, transformed", "augment")):
        mentioned = True
    if mentioned and not a0_ids:
        a0_ids = gazetteer_ids[:4]
    if not mentioned:
        return AugmentFacts(mentioned=False)
    bound, _ = _unwrap(payload, "A1")
    relic, _ = _unwrap(payload, "A2")
    bound_val = _pick(bound, {"bound", "not_bound", "n/a", "not_mentioned"})
    if "altered, arranged, transformed" in blob or "remix" in blob:
        if bound_val not in {"not_bound"}:
            bound_val = "bound"
    if bound_val == "n/a":
        bound_val = "not_mentioned"
    relic_val = _pick(
        relic, {"yes", "no", "conditional", "n/a", "not_mentioned", "true", "false"}
    )
    if relic_val == "n/a":
        relic_val = "not_mentioned"
    return AugmentFacts(
        mentioned=True,
        bound=bound_val,
        relicense=relic_val,
        evidence_ids=list(a0_ids or gazetteer_ids[:4]),
    )


def _parse_stat(payload: dict, evidence: list[EvidenceItem]) -> StatFacts:
    s0, s0_ids = _unwrap(payload, "S0")
    gazetteer_ids = [e.id for e in evidence if e.topic in STAT_TOPICS]
    blob = " ".join(
        (e.quote or "").lower() for e in evidence if e.topic in STAT_TOPICS
    )
    mentioned = _pick(s0, {"yes", "no"}, default="no") == "yes"
    if any(
        cue in blob
        for cue in (
            "statistical analysis",
            "statistical use",
            "computational use",
            "not itself a linguistic resource",
            "not a linguistic resource",
        )
    ):
        mentioned = True
    if mentioned and not s0_ids:
        s0_ids = gazetteer_ids[:4]
    if not mentioned:
        return StatFacts(mentioned=False)
    bound, _ = _unwrap(payload, "S1")
    relic, _ = _unwrap(payload, "S2")
    bound_val = _pick(bound, {"bound", "not_bound", "n/a", "not_mentioned"})
    if "not itself a linguistic resource" in blob or "not a linguistic resource" in blob:
        bound_val = "not_bound"
    if bound_val == "n/a":
        bound_val = "not_mentioned"
    relic_val = _pick(
        relic, {"yes", "no", "conditional", "n/a", "not_mentioned", "true", "false"}
    )
    if relic_val == "n/a":
        relic_val = "not_mentioned"
    return StatFacts(
        mentioned=True,
        bound=bound_val,
        relicense=relic_val,
        evidence_ids=list(s0_ids or gazetteer_ids[:4]),
    )
