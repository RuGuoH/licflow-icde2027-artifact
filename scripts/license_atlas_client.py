"""
LicenseAtlas 本地全文读取（需先运行 sync_license_atlas.py 生成 bodies/）。
数据源: https://github.com/morningD/license.atlas
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_ATLAS_DIR = os.path.join(os.path.dirname(__file__), "license_atlas")

_INDEX: Optional[dict] = None

# 与 work_resolver_helpers 对齐的常见别名
_BUILTIN_ALIASES = {
    "mit license": "mit",
    "mit": "mit",
    "apache 2.0": "apache-2.0",
    "apache-2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache 2": "apache-2.0",
    "apache": "apache-2.0",
    "gpl-3.0": "gpl-3.0",
    "gpl 3": "gpl-3.0",
    "gplv3": "gpl-3.0",
    "cc-by-sa-4.0": "cc-by-sa-4.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc-by-nc-4.0": "cc-by-nc-4.0",
    "cc0": "cc0-1.0",
    "cc0-1.0": "cc0-1.0",
    "unlicense": "unlicense",
    "bsd-3-clause": "bsd-3-clause",
    "creativeml-openrail-m": "creativeml-openrail-m",
    "llama2": "llama-2-community-license-agreement",
    "llama 2": "llama-2-community-license-agreement",
    "llama-license-2023": "llama-license-2023",
    "llama license agreement 2023": "llama-license-2023",
}


def _enabled() -> bool:
    return os.getenv("LICENSE_ATLAS_ENABLED", "true").lower() in ("1", "true", "yes")


def _atlas_dir() -> str:
    return os.getenv("LICENSE_ATLAS_DIR", _DEFAULT_ATLAS_DIR)


def _bodies_dir() -> str:
    return os.path.join(_atlas_dir(), "bodies")


def _index_path() -> str:
    return os.path.join(_atlas_dir(), "index.json")


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _slugify(license_name: str) -> str:
    s = (license_name or "").strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _is_placeholder_license_name(license_name: str) -> bool:
    low = (license_name or "").lower()
    return low.endswith("_license") or low.endswith("-license")


def load_index(force: bool = False) -> Optional[dict]:
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX
    path = _index_path()
    if not os.path.isfile(path):
        logger.debug("LicenseAtlas index not found: %s (run sync_license_atlas.py)", path)
        _INDEX = None
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            _INDEX = json.load(f)
        return _INDEX
    except Exception as e:
        logger.warning("Failed to load LicenseAtlas index %s: %s", path, e)
        _INDEX = None
        return None


def resolve_slugs(license_name: str) -> List[str]:
    """license_name → 候选 Atlas slug（去重保序）。"""
    if not license_name or not str(license_name).strip():
        return []
    if _is_placeholder_license_name(license_name):
        return []

    name = str(license_name).strip()
    seen: set[str] = set()
    out: List[str] = []

    def add(slug: Optional[str]) -> None:
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)

    index = load_index()
    if index:
        by_slug = index.get("by_slug") or {}
        low_slug = name.lower()
        if low_slug in by_slug:
            add(low_slug)

        by_spdx = index.get("by_spdx_id") or {}
        if name in by_spdx:
            add(by_spdx[name])
        for key, slug in by_spdx.items():
            if key.lower() == name.lower():
                add(slug)

        aliases = index.get("aliases") or {}
        low = name.lower()
        if low in aliases:
            add(aliases[low])
        norm = _normalize_key(name)
        for alias_key, slug in aliases.items():
            if _normalize_key(alias_key) == norm:
                add(slug)

    for alias_key, slug in _BUILTIN_ALIASES.items():
        if alias_key == low or _normalize_key(alias_key) == _normalize_key(name):
            add(slug)

    add(_slugify(name))
    if name != name.lower():
        add(_slugify(name.lower()))

    return out


def read_body(slug: str) -> Optional[str]:
    path = os.path.join(_bodies_dir(), f"{slug}.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None
    except Exception as e:
        logger.debug("Failed to read LicenseAtlas body %s: %s", path, e)
        return None


def atlas_is_ready() -> bool:
    """index 与 bodies 目录均存在（不保证每条许可都有 body）。"""
    return os.path.isfile(_index_path()) and os.path.isdir(_bodies_dir())


def fetch_from_atlas(license_name: str) -> Optional[str]:
    if not _enabled():
        return None
    if not atlas_is_ready():
        logger.debug(
            "LicenseAtlas not ready (missing index or bodies/). "
            "Run: make sync-atlas"
        )
        return None

    for slug in resolve_slugs(license_name):
        text = read_body(slug)
        if text:
            logger.info(
                "LicenseAtlas hit: %s → slug=%s (%d chars)",
                license_name,
                slug,
                len(text),
            )
            return text

    logger.debug("LicenseAtlas miss for license_name=%s slugs=%s", license_name, resolve_slugs(license_name))
    return None
