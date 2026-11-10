"""OpenAI 兼容 /embeddings 客户端。复用 Agent 的 OPENAI_BASE_URL 网关。"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def resolve_embed_credentials() -> tuple[str, str]:
    """与 llm_license_helper / agent.llms 同一套兼容网关。"""
    from scripts.llm_license_helper import _default_openai_credentials

    key, base = _default_openai_credentials()
    if not key or not base:
        raise RuntimeError("缺少 OPENAI_API_KEY / OPENAI_BASE_URL，无法调用 embeddings")
    return key, base.rstrip("/")


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBED_MODEL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    batch_size: int = 64,
    timeout: int = 120,
) -> list[list[float]]:
    """按批调用 /embeddings，返回与 texts 等长的向量列表。"""
    if not texts:
        return []
    key = api_key
    base = api_base
    if not key or not base:
        key, base = resolve_embed_credentials()
    url = f"{base.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    out: list[Optional[list[float]]] = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        cleaned = [t if (t or "").strip() else " " for t in chunk]
        resp = requests.post(
            url,
            headers=headers,
            json={"model": model, "input": cleaned},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"embeddings HTTP {resp.status_code}: {(resp.text or '')[:400]}"
            )
        data = resp.json()
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        if len(rows) != len(cleaned):
            raise RuntimeError(
                f"embeddings 返回条数不匹配: got {len(rows)} want {len(cleaned)}"
            )
        for i, row in enumerate(rows):
            emb = row.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"embeddings 缺 embedding 字段 index={start + i}")
            out[start + i] = emb
        logger.info(
            "embed batch %d-%d model=%s dim=%d",
            start,
            start + len(cleaned) - 1,
            model,
            len(rows[0]["embedding"]),
        )
    return [v for v in out if v is not None]
