import copy
import logging
import requests
import re
import threading
import time
import yaml
import os
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from scripts.api_license_fetcher import fetch_license_from_api
from scripts.license_atlas_client import fetch_from_atlas

# 两阶段审计自动修复的最大重试次数
MAX_METADATA_RETRY = 3
MAX_TERMS_RETRY = 3

# 审计 evaluate 返回 None（API/YAML 失败）时的重试次数，不计入 fix 轮次
MAX_AUDIT_EVAL_RETRY = 3

# LLM HTTP 调用重试（应对断连、超时、5xx 等瞬时故障）
LLM_CALL_MAX_RETRIES = 3
LLM_CALL_RETRY_BACKOFF_S = 2.0

# 未知许可建模后端：legacy = 现有两阶段 generate-audit-fix（默认）；qa = 抽取+问卷（开发中）
LICENSE_MODELING_PIPELINE_LEGACY = "legacy"
LICENSE_MODELING_PIPELINE_QA = "qa"
_QA_PIPELINE_ALIASES = frozenset({"qa", "extract_qa", "v2"})


def resolve_license_modeling_pipeline(value: Optional[str] = None) -> str:
    """读取 LICENSE_MODELING_PIPELINE；未知值回退 legacy，保证默认行为不变。"""
    raw = (value if value is not None else os.getenv("LICENSE_MODELING_PIPELINE") or "").strip().lower()
    if not raw:
        return LICENSE_MODELING_PIPELINE_LEGACY
    if raw in _QA_PIPELINE_ALIASES:
        return LICENSE_MODELING_PIPELINE_QA
    if raw == LICENSE_MODELING_PIPELINE_LEGACY:
        return LICENSE_MODELING_PIPELINE_LEGACY
    logging.warning("未知 LICENSE_MODELING_PIPELINE=%r，回退为 legacy", raw)
    return LICENSE_MODELING_PIPELINE_LEGACY


def _default_openai_credentials() -> tuple[str, str]:
    """与 agent/llms.py 对齐：env 优先，否则用 MODEL_CONFIGS 默认值。"""
    base = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    if base and key:
        return key, base.rstrip("/")
    try:
        from agent.llms import MODEL_CONFIGS

        cfg = MODEL_CONFIGS["openai_4o_min"]
        return key or cfg["api_key"], (base or cfg["base_url"]).rstrip("/")
    except Exception:
        return key or "", (base or "https://api.openai.com/v1").rstrip("/")


# 走 OPENAI_BASE_URL 兼容网关的 preset（非 DeepSeek 官方 API）
_OPENAI_COMPAT_PRESETS = frozenset({"openai"})

# LICENSE_LLM_MODEL 别名 → preset 键
_LICENSE_LLM_PRESET_ALIASES: Dict[str, str] = {
    "deepseek_v4_pro": "deepseek-pro",
    "deepseek-v4-pro": "deepseek-pro",
}


def normalize_license_llm_preset(model: str) -> str:
    key = (model or "").lower()
    return _LICENSE_LLM_PRESET_ALIASES.get(key, key)


_DEEPSEEK_OFFICIAL_BASE = "https://api.deepseek.com/v1"


def build_supported_models() -> Dict[str, Dict[str, str]]:
    """许可证建模 LLM preset。deepseek / deepseek-pro 走官网；openai 复用 OPENAI_BASE_URL。"""
    _, openai_base = _default_openai_credentials()
    return {
        "deepseek": {
            "model": "deepseek-v4-flash",
            "api_base": _DEEPSEEK_OFFICIAL_BASE,
        },
        "deepseek-pro": {
            "model": "deepseek-v4-pro",
            "api_base": _DEEPSEEK_OFFICIAL_BASE,
        },
        "qianwen": {
            "model": "qwen3-max",
            "api_base": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        },
        "openrouter": {
            "model": "google/gemini-2.5-pro",
            "api_base": "https://openrouter.ai/api/v1",
        },
        "openai": {
            "model": "gpt-4o-mini",
            "api_base": openai_base,
        },
    }


def resolve_license_llm_api_key(model: str, api_key: Optional[str] = None) -> Optional[str]:
    """openai preset 优先 OPENAI_API_KEY（或 Agent 默认），其余用 LICENSE_LLM_API_KEY。"""
    preset = normalize_license_llm_preset(model)
    if preset in _OPENAI_COMPAT_PRESETS or preset.startswith("gpt-"):
        key, _ = _default_openai_credentials()
        return key or api_key
    return api_key


class LLMLicenseHelper:
    """
    使用 LLM 对未知许可证进行结构化建模与审计的辅助类。
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-v4-flash", api_base: Optional[str] = None,
                 github_token: Optional[str] = None):
        # 支持的模型类型 (DeepSeek, 通义千问, OpenRouter, OpenAI)
        self.supported_models = build_supported_models()
        self.github_token = github_token

        preset = normalize_license_llm_preset(model)
        if preset in self.supported_models:
            self.model_type = preset
            self.model = self.supported_models[preset]["model"]
            self.api_base = api_base or self.supported_models[preset]["api_base"]
        else:
            self.model_type = "custom"
            self.model = model
            _, default_base = _default_openai_credentials()
            self.api_base = api_base or default_base

        self.api_key = resolve_license_llm_api_key(model, api_key)
        self.license_cache: Dict[str, str] = {}
        self.last_llm_meta: Dict = {}
        self._llm_meta_local = threading.local()

        # 保存 LLM 输出的目录（用于调试与审计）
        self.llm_output_dir = os.path.join(os.getcwd(), "llm_outputs")
        os.makedirs(self.llm_output_dir, exist_ok=True)
        logging.info("LLM 输出将保存到: %s", self.llm_output_dir)

    def _remember_llm_meta(self, meta: Optional[Dict]) -> None:
        """同一 helper 上的并发调用各记各的返回信息。"""
        stored = dict(meta or {})
        self.last_llm_meta = stored
        self._llm_meta_local.meta = stored

    def current_llm_meta(self) -> Dict:
        meta = getattr(self._llm_meta_local, "meta", None)
        if isinstance(meta, dict):
            return meta
        return dict(self.last_llm_meta or {})

    # --------- 工具函数：保存输出 ---------

    def save_llm_output(self, license_name: str, llm_response: str, output_type: str = "license_analysis") -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_license_name = re.sub(r"[^a-zA-Z0-9._-]", "_", license_name)
            filename = f"{output_type}_{safe_license_name}_{timestamp}.txt"
            filepath = os.path.join(self.llm_output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 许可证名称: {license_name}\n")
                f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 输出类型: {output_type}\n\n")
                f.write(llm_response)
            logging.info("LLM 输出已保存到: %s", filepath)
            return filepath
        except Exception as e:
            logging.error("保存 LLM 输出失败: %s", e)
            return ""

    def save_revision_for_inspection(
        self,
        license_name: str,
        data: Dict,
        revision_type: str,
        attempt: int,
        extra_label: str = "",
    ) -> str:
        """
        将每次自动修正后的内容保存为 YAML 文件，供人工检查。
        """
        try:
            revisions_dir = os.path.join(self.llm_output_dir, "revisions")
            os.makedirs(revisions_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", license_name)
            label = f"_{extra_label}" if extra_label else ""
            filename = f"{revision_type}_{safe_name}_attempt{attempt}{label}_{timestamp}.yaml"
            filepath = os.path.join(revisions_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 许可证: {license_name}\n")
                f.write(f"# 修订类型: {revision_type}\n")
                f.write(f"# 第 {attempt} 次修正 | 保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(yaml.dump(data, allow_unicode=True, sort_keys=False))
            logging.info("修订内容已保存供检查: %s", filepath)
            print(f"修订内容已保存供检查: {filepath}")
            return filepath
        except Exception as e:
            logging.error("保存修订内容失败: %s", e)
            return ""

    # --------- 通用 LLM 调用 ---------

    @staticmethod
    def _is_retryable_llm_exception(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        )

    @staticmethod
    def _is_retryable_http_status(status_code: int) -> bool:
        return status_code in (429, 502, 503, 504)

    def _post_llm_once(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
        extra: Optional[Dict] = None,
    ) -> requests.Response:
        if self.model_type == "qianwen":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "input": {"messages": messages},
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "result_format": "message",
                },
            }
            response = requests.post(self.api_base, headers=headers, json=data, timeout=timeout)
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if extra:
                data.update(extra)
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout,
            )
        return response

    def _parse_llm_response(self, response: requests.Response) -> Optional[str]:
        data = response.json()
        meta: Dict = {
            "status": response.status_code,
            "finish_reason": None,
            "content_chars": 0,
            "reasoning_chars": 0,
        }
        if self.model_type == "qianwen":
            content = (
                data.get("output", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            ).strip()
            meta["content_chars"] = len(content)
            self._remember_llm_meta(meta)
            return content
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        meta["finish_reason"] = choice.get("finish_reason")
        meta["content_chars"] = len(content)
        meta["reasoning_chars"] = len(reasoning)
        meta["usage"] = data.get("usage")
        self._remember_llm_meta(meta)
        return content

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 180,
        extra: Optional[Dict] = None,
    ) -> Optional[str]:
        if not self.api_key:
            logging.error("API key not provided for LLM license helper")
            return None

        self._remember_llm_meta({})
        for attempt in range(LLM_CALL_MAX_RETRIES):
            try:
                response = self._post_llm_once(
                    messages, temperature, max_tokens, timeout, extra=extra
                )
                if response.status_code != 200:
                    self._remember_llm_meta({
                        "status": response.status_code,
                        "http_error": True,
                        "body_head": (response.text or "")[:400],
                    })
                    if (
                        self._is_retryable_http_status(response.status_code)
                        and attempt < LLM_CALL_MAX_RETRIES - 1
                    ):
                        wait = LLM_CALL_RETRY_BACKOFF_S * (2 ** attempt)
                        logging.warning(
                            "LLM API 返回 %s，%.0fs 后进行第 %d 次重试",
                            response.status_code,
                            wait,
                            attempt + 2,
                        )
                        time.sleep(wait)
                        continue
                    logging.error(
                        "LLM API 请求失败: %s - %s",
                        response.status_code,
                        response.text,
                    )
                    return None

                content = self._parse_llm_response(response)
                if attempt > 0:
                    logging.info("LLM 请求在第 %d 次重试后成功", attempt + 1)
                return content
            except Exception as e:
                self._remember_llm_meta({"exception": f"{type(e).__name__}: {e}"})
                if self._is_retryable_llm_exception(e) and attempt < LLM_CALL_MAX_RETRIES - 1:
                    wait = LLM_CALL_RETRY_BACKOFF_S * (2 ** attempt)
                    logging.warning(
                        "调用 LLM 失败 (%s)，%.0fs 后进行第 %d 次重试",
                        e,
                        wait,
                        attempt + 2,
                    )
                    time.sleep(wait)
                    continue
                logging.error("调用 LLM 失败: %s", e)
                return None

        return None

    def _legacy_cap(self, stage: str, kind: str, default: int) -> tuple[int, int, Optional[Dict]]:
        """实验运行可在实例上设置 legacy_limits，覆盖长度、超时和思考开关。未设置时保持原参数。"""
        limits = getattr(self, "legacy_limits", None) or {}
        if kind == "fix":
            cap_key = "fix_metadata" if stage == "metadata" else "fix_terms"
        elif kind == "audit":
            cap_key = "audit"
        else:
            cap_key = "generate"
        cap = int(limits.get(cap_key, default))
        timeout = int(limits.get("timeout", 180))
        extra = limits.get("extra") if isinstance(limits, dict) else None
        if isinstance(extra, dict):
            extra = dict(extra)
        else:
            extra = None
        return cap, timeout, extra

    def _record_legacy_call(
        self,
        *,
        stage: str,
        kind: str,
        max_tokens: int,
        started: float,
        text: Optional[str],
    ) -> None:
        records = getattr(self, "legacy_call_records", None)
        if not isinstance(records, list):
            return
        meta = dict(self.last_llm_meta or {})
        usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details")
        if not isinstance(details, dict):
            details = {}
        reasoning = details.get("reasoning_tokens")
        if reasoning is None:
            reasoning = usage.get("reasoning_tokens")
        if reasoning is None:
            reasoning = 0
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if total is None and isinstance(prompt, int) and isinstance(completion, int):
            total = prompt + completion
        parsed_ok = False
        if text and str(text).strip():
            body = re.sub(r"^```(yaml)?\s*|\s*```$", "", str(text).strip(), flags=re.MULTILINE)
            try:
                parsed_ok = isinstance(yaml.safe_load(body), (dict, list))
            except Exception:
                parsed_ok = False
        same = [row for row in records if row.get("stage") == stage and row.get("kind") == kind]
        records.append(
            {
                "stage": stage,
                "kind": kind,
                "attempt": len(same) + 1,
                "max_tokens": max_tokens,
                "parsed_ok": parsed_ok,
                "finish_reason": meta.get("finish_reason"),
                "wall_s": round(time.perf_counter() - started, 3),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning,
                "total_tokens": total,
                "content_chars": meta.get("content_chars"),
                "status": meta.get("status"),
            }
        )

    def _tag_last_legacy(self, stage: str, kind: str, **fields) -> None:
        records = getattr(self, "legacy_call_records", None)
        if not isinstance(records, list):
            return
        for row in reversed(records):
            if row.get("stage") == stage and row.get("kind") == kind and not row.get("_tagged"):
                row.update(fields)
                row["_tagged"] = True
                return

    def _legacy_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        stage: str,
        kind: str,
        max_tokens: int,
        temperature: float = 0.1,
    ) -> Optional[str]:
        cap, timeout, extra = self._legacy_cap(stage, kind, max_tokens)
        started = time.perf_counter()
        text = self.call_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=cap,
            timeout=timeout,
            extra=extra,
        )
        self._record_legacy_call(
            stage=stage,
            kind=kind,
            max_tokens=cap,
            started=started,
            text=text,
        )
        return text

    # --------- 许可证文本获取 ---------

    def _read_license_text_from_raw(self, license_name: str) -> Optional[str]:
        """从 scripts/license_raw/ 读取许可全文（精确匹配优先，其次子串匹配）。"""
        try:
            base_dir = os.path.join(os.path.dirname(__file__), "license_raw")
            if not os.path.exists(base_dir):
                return None

            license_files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]
            processed_target = re.sub(r"[^a-zA-Z0-9]", "", license_name.lower())
            license_path = None

            for filename in license_files:
                name = os.path.splitext(filename)[0]
                processed = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
                if processed == processed_target:
                    license_path = os.path.join(base_dir, filename)
                    break

            if not license_path:
                for filename in license_files:
                    name = os.path.splitext(filename)[0]
                    processed = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
                    if processed_target in processed or processed in processed_target:
                        license_path = os.path.join(base_dir, filename)
                        break

            if license_path and os.path.exists(license_path):
                with open(license_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logging.debug("Failed to read %s from local license_raw: %s", license_name, e)
        return None

    def fetch_license_text(self, license_name: str) -> Optional[str]:
        if license_name in self.license_cache:
            return self.license_cache[license_name]

        text = self._read_license_text_from_raw(license_name)
        if text:
            self.license_cache[license_name] = text
            return text

        text = fetch_from_atlas(license_name)
        if text:
            self.license_cache[license_name] = text
            return text

        urls = [
            f"https://raw.githubusercontent.com/spdx/license-list-data/main/text/{license_name}.txt",
            f"https://opensource.org/licenses/{license_name}.txt",
            f"https://www.gnu.org/licenses/{license_name}.txt",
        ]
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    text = resp.text.strip()
                    self.license_cache[license_name] = text
                    return text
            except Exception as e:
                logging.debug("Failed to fetch %s from %s: %s", license_name, url, e)

        logging.warning("Could not fetch license text for %s", license_name)
        return None

    # --------- 元数据 & terms 生成---------

    def analyze_license_with_llm_data(self, license_name: str, license_text: str) -> Optional[Dict]:
        """使用 LLM 分析许可证并返回符合 licenses_description.yml 格式的元数据（不含 terms）。"""
        if not self.api_key:
            logging.error("API key not provided. Please set it when initializing LLMLicenseHelper")
            return None
        try:
            prompt = f"""
            你是一位专业的许可证分析专家，精通开源软件、开放数据、AI 模型及内容许可证的法律与技术细节。请严格依据以下规则分析给定的许可证：

            许可证名称: {license_name}
            许可证文本:
            {license_text}

            请仅输出符合 YAML 格式的许可证元数据，不要包含任何解释、注释、编号列表或中文说明。不要使用 Markdown 代码块（如 ```yaml）。若仅提供许可证名称而无文本，请基于权威知识（如 SPDX、OSI、Creative Commons 官方定义）填写；若有文本，则所有字段必须严格基于文本内容，不得凭许可证名称或家族补写，不得补充文本未提及的信息。
            输出 YAML 时，字段顺序必须严格按照本提示词出现的顺序
            full_name: 完整的许可证官方名称（不得添加用途说明如 "Data License"）
            short_id: 许可证的短标识符（必须与输入的 license_name 完全一致）
            url: 许可证的官方 URL。原文给出就填写；原文没有则填 null，不要编造
            version: 许可证版本号；若无法确定，设为 null            
            available: true                                                

            categories: [public, software, data, model, proprietary, permissive, copyleft, disclose, auto-relicensing]
            # 必须从上述列表中选择适用项。注意：
            #   - "public": 只要该许可证是向不特定公众公开发布的（即 Open Source 或 Shared Source），许可证允许社会大众使用，而非针对特定商业伙伴的定制协议，均添加此项          
            #   - "permissive"：宽松许可（如 MIT、Apache-2.0），根据实际内容判断，不得通过许可证名称简单臆测            
            #   - "copyleft"：copyleft许可（如 GPL、CC-BY-SA）。如果修改或基于某个作品创作衍生作品，并分发该衍生作品，必须使用相同或兼容的许可证来授权整个衍生作品。            
            #   - "disclose"：意味着许可作品要提供源代码（如 GPL 要求提供源代码，AFL-3.0要求原始许可人提供源码）            
            #   - "auto-relicensing"：衍生作品必须以相同或兼容许可证发布（如 CC-BY-SA、GPL)            
            #   - "software"：此许可适用于software           
            #   - "data"：此许可适用于data            
            #   - "model"：此许可适用于model  
            #   - 这里的"适用于"是指此许可本身设计时考虑了software, data, model中的某种或多种，不用考虑现实是否有人用作它用，比如经常有人使用Apache-2.0做model许可，但其实它在设计时并没有考虑model          
            #   - "proprietary"（专有）与 "permissive"（宽松）互斥，不能同时存在 
            #   - "public" 和 "proprietary" 可以同时出现（例如公开发布文本/源码的专有许可证）
            #   - 括号中的许可证名称只帮助理解类别含义，不是查找表。不得因为当前名称与例子相近就自动勾选

            labels: []            
            # 可选标签（仅当明确适用时添加）：            
            #   "OSI Approved", "Public Domain", "GNU", "Creative Commons", "Open Data Commons",            
            #   "Microsoft", "OpenRAIL", "Responsible AI", "Use Restriction", "Meta"            

            rights: [use, modify, merge, redistribute, sublicense, commercial_use, patent_use, copyright]             
            # 仅列出许可证明确授予的权利（这里的权利是指不在分发状态下的权利。例如某许可证允许分发未修改的原件、但不允许分发改编版，则 rights 保留 redistribute 与 modify，coverage 只列 duplicate；不要靠删掉 redistribute 来表示禁止改编）仅使用上述值。注意：
            #   - "merge" 指将原作品与其他材料组合成集合，只要未修改原内容。仅当原文授予了组合、聚合或同类含义的权利时才添加
            #   - "patent_use"（专利使用权）指的是被许可人使用该专利技术的权利。仅当原文明确授予时才列入 rights；未提及则不列入
            #   - "sublicense" 是指被许可人（Licensee）拥有将自己获得的权利，以自己的名义再次授权给第三方（Sublicensee）的法律权力。只有当许可证文本中明确出现 "sublicense"、"sub-license" 或 "right to grant sublicenses" 等词汇时，才能将此项归入 rights
            #   - 只能包含上述已列出内容，不得新加词汇                        
            #   - 再分发（redistribute）和再许可（sublicense）不同。redistribute 是指把原作品（及原协议）传给别人；sublicense 是指被许可人有权针对原作品签发一份新的许可协议。redistribute 仅当原文明确允许再分发时列入
            #   - 若许可证声明放弃版权（如 Unlicense、CC0），则添加 "copyright"
            #   - 不得包含未授权的行为（rights 和 reserved_rights 无交集）。原文禁止的行为不得写入 rights，例如原文禁止商业使用时不得包含 commercial_use


            reserved_rights: [trademark_use, copyright, patent, trademark, sublicense, commercial_use, patent_use, redistribute]
            # 仅列出许可证明确保留的法律权利，仅使用上述值。重要规则：
            #   - patent（专利权本身）指的是对某项发明所拥有的完整的专利所有权
            #   - patent_use（专利使用权）指的是被许可人使用该专利技术的权利
            #   - trademark（商标权本身）
            #   - trademark_use（商标使用权）指是否允许被许可人使用该商标。如果许可证明示不授予商标使用权在reserved_rights列出trademark_use
            #   - 不能包含预定义列表内容之外的词汇。特别注意，严禁擅自添加如 modify, copy, share 等未在预定义列表中的行为
            #   - 仅当原文明确保留、拒绝授予或禁止某项权利时才列入。未提及的权利不要因为许可家族而补写
            #   - 例如许可证说 “you may not use for commercial purposes”，则 reserved_rights 必须包含 "commercial_use"
            #   - 若许可证说 “no right to sublicense”，则 reserved_rights 必须包含 "sublicense"
            #   - 只有当原文明确不授予 patent_use（专利使用权）时，才在 reserved_rights 中列入 patent_use
            #   - rights 和 reserved_rights 应无交集（disjoint）
            #   - "sublicense" 只有当许可证文本中明确不允许时才能归入 reserved_rights


             rights_prefix: [perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable, non-transferable, non-sublicensable, revocable, sublicensable, any-person]
            # 描述所授予权利的性质前缀，仅使用上述值。此部分严格遵循许可证原文，未明确提及的不要臆测。 
            # - perpetual: 指授权一旦授予，在满足许可证条款的前提下永久有效，不会因时间到期而自动终止, 除非有明文写出（irrevocable不等同于perpetual），否则不得添加
            # - non-transferable: 被授权方不能将所获权利转授给第三方
            # - 注意区分：no-charge：排除任何形式的初始许可费，例如 free of charge；royalty-free：排除基于使用量、收入、分发数量等的后续分成或计费。两者不互斥，具体情况遵循许可证原文。

            coverage: [duplicate, derivative, modification, translation]
            # 表示哪些类型的作品可以被合法再分发：
            #   - "duplicate"：原样复制。原文允许复制或分发副本时列入；原文明确禁止复制时不列入
            #   - "derivative/modification/translation"：仅当原文允许分发修改版、改编版或翻译版时才可能包含
            #   - "derivative"：表示新作品与原作品在法律上无法分离，新作品受原许可证的条款约束
            #   - "modification"：原文把产物描述为对内容的改动，且该改动仍触发与衍生作品相同的义务
            #   - "translation"：原文把产物描述为语言或格式上的转换。不要因为许可名称而选用
            #   - 若原文声明不保留任何权利，所有结果同等对待、没有任何再分发限制，则 coverage 列表置空

            redistribute: [include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]
            # 只列入原文明确要求的再分发义务，不要因为允许再分发就把下面五项全部写入：
            #   - include_notice：保留版权声明、归因、免责声明，或指向本许可的说明
            #   - include_license：随附本许可全文，或要求标明本许可并给出许可文本或 URI
            #   - state_changes：必须说明作品是否被修改，或保留修改标记。仅当原文有此要求，且 coverage 含 derivative、modification 或 translation 时列入
            #   - include_use_restriction：再分发时必须保留原文列出的使用限制
            #   - include_runtime_restriction：再分发或提供服务时必须保留运行时限制
            # 原文同时要求保留声明并随附许可文本时，include_notice 与 include_license 都要列入

            compat: []
            # 仅当许可证原文明确声明与某许可证兼容时填写。不得按许可家族补写系列内部兼容名单

            incompat: []
            # 仅当许可证原文明确声明不兼容某些许可证时填写。不得按许可家族补写不兼容名单

            【最终校验清单】
            在输出前，请逐项确认：
            1. reserved_rights 中不含 modify/copy/share/sell 等行为动词。
            2. rights 中的 "sublicense" 仅在许可证明文授权时出现。
            3. rights = 被明确授予的使用权，reserved_rights = 明确未授予或禁止的权利，二者应无交集（disjoint）。这里的无交集仅指**完全相同的词汇字符串**不得同时出现在两边（例如 commercial_use 不能同时在两边出现），无需过度引申。
            4. 若 coverage 不包含 derivative/modification，则 redistribute 不得包含 state_changes。
            5. 不得按许可证名称或家族补写 rights、reserved_rights、coverage、compat、incompat。
            6. 所有字段值必须来自预定义枚举
            """
            messages = [
                {"role": "system", "content": "你是一位专门分析软件和数据许可证的专家。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self._legacy_chat(
                messages, stage="metadata", kind="generate", max_tokens=20000
            )
            if not llm_response:
                return None
            self.save_llm_output(license_name, llm_response, "license_analysis_data")
            yaml_content = re.sub(r"^```(yaml)?\s*|\s*```$", "", llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r"---\s*([\s\S]*?)\s*---", yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            return yaml.safe_load(yaml_content)
        except Exception as e:
            logging.error("Failed to analyze license with LLM: %s", e)
            return None

    def analyze_license_with_llm_terms(
        self, license_name: str, license_text: str, license_data_no_terms: str
    ) -> Optional[Dict]:
        """在已有元数据基础上，使用 LLM 生成 terms 并返回完整 license_data。"""
        if not self.api_key:
            return None
        try:
            prompt = f"""
            你是一位专业的许可证分析专家，精通开源软件、开放数据、AI 模型及内容许可证的法律与技术细节。

            许可证名称: {license_name}
            许可证文本:
            {license_text}
            许可证元数据:
            {license_data_no_terms}
            # 任务：作为许可证分析专家，你需要在许可证元数据的基础上生成terms条目，对于输入的许可证元数据，请完全原样保留，绝对不要在这一步修改它的内容，只在末尾增加terms结构。
            请严格依据以下规则分析给定的许可证文本
            请仅输出符合 YAML 格式的增加了terms的许可证元数据，不要包含任何解释、注释、编号列表或中文说明。不要使用 Markdown 代码块（如 ```yaml）。若仅提供许可证名称而无文本，请基于权威知识（如 SPDX、OSI、Creative Commons 官方定义）填写；若有文本，则所有字段必须基于文本内容，不得凭许可证名称或家族补写活动或结果。
            输出内容必须是完整的，包括**许可证元数据**和**terms**

            ## 分析核心原则：所有分析的最终依据必须是许可证文本本身。任何结论都要能指回原文措辞，不得凭许可证名称或家族推断。
            # 以下描述中的原作品是指使用当前分析许可证的 data、software 或 model。
            # 阅读时把下面几件事分开。都只看本许可证原文，不要套用其他许可的常见答案：
            # 1. 使用限制或可接受使用政策不是同许可传染。只有原文要求该活动的结果必须以本许可、或原文点名的兼容许可再发布时，该项 copyleft 才为 true。
            # 2. 输出本身是否仍受本许可约束，与使用该输出时是否还要遵守使用限制，是两件事。原文声明不对输出主张权利时，不要把 generate 的 result 写成 derivative；若同时要求输出的使用遵守使用限制，把该限制写入 restrictions。
            # 3. 原文若把「仅仅放在一起、仍然可分离」和「改动后融成一部作品」分开规定，前者写入 combine，result 按原文；后者写入 modify 或 amalgamate，result 按原文。原文没有这种区分时，不要自行拆成两种结果。
            # 4. 原文没有按源码、二进制或网络服务规定不同后果时，不要为同一活动拆出互相矛盾的 forms。
            # 5. train 只在原文讨论训练、微调或把权重、参数迁移到另一模型时列入。generate 只在原文讨论运行产物或输出时列入。embed 只在原文讨论嵌入、翻译或改编材料时列入。只出现「衍生作品」，不足以写入 train 或 generate。

            # terms 编写详细规则：
            - usages: []
                #只列出原文明确涉及的活动。允许、禁止或附带限制都算涉及。原文完全没有提及的活动不要补写，也不要因为许可证名称或家族补上训练、生成等活动。
                #一个活动可以单独列出为一组。如果活动具有相同 result, forms, relicense, restrictions 和 copyleft 属性，可以分在一组，合并为一组时 keywords 合并在一个列表中。
                #原文已经写明产物法律后果时，result 不得为 NODEF。原文讨论了该活动但没有说明产物是否受约束、能否另行许可时，用 NODEF，不要猜。
                #以下活动在不同许可中的用词可能不同，要根据活动含义理解，而不只是寻找某个单词。只有原文措辞对得上该活动含义时才列入。
                #例如原文不允许利用输出改进模型，才是对 distill 的限制。不得仅因本许可用于模型、或原文出现“衍生作品”，就写入 train。
                #合法值如下（只可以出现以下活动，未涉及的不要列出）
                #copy  #新作品是原作品的复制品
                #use  #单纯的使用，这里不产生新作品，后续分析时新作品等同原作品
                #modify #新作品modify了原作品
                #train #新作品基于原作品训练而成，例如新作品由原作品微调而成
                #combine #将多部作品(同种类型作品的组合，例如数据和数据的组合, 原作品是其中之一，没有对原作品进行修改, 只是组合）组合起来构建一部新作品。或者将原作品放在另外一个大的作品集合中（同类型作品）。组合后的作品整体是新作品。仅当原文涉及组合、聚合或同类安排时才列出
                #combine_mix #将多个作品组合以构建一个新作品（原作品是其中之一），由此产生的新作品是一种混合类型的组合（可能包括data, model, software多种类型作品进行组合, 没有对原作品进行修改，只是组合）。或者将原作品放在另外一个大的作品集合中（混合类型作品）。组合后的作品整体是新作品。
                #amalgamate #将多个作品进行融合（原作品是其中之一），生成无法再分离的新作品。或者将原作品融入另一个作品中。融合后的作品整体是新作品。
                #augment #在不改变数据核心标签含义的前提下，对原始数据进行一系列随机或预设的变换，从而得到更多、更多样化的新样本，新作品是原作品（通常是data）增强产生的
                #distill # 模型蒸馏，Distill knowledge from old models to new models。新作品对原作品进行了蒸馏
                #generate #原作品生成新作品。仅当原文明确规范输出、生成结果或同类产物时才列入。不要因为许可名称或其他家族的惯例写入 generate
                #embed # Embed works (corpus, image or other data samples) using aux_works (model or algorithm)，将原作品（通常是data）利用 model 或者 algorithm 嵌入到新作品
                #stat #对数据进行统计，新作品是原作品的统计分析数据

            forms: [raw, binary, saas]
                # 三选一或多选：
                # - raw: 原始格式（源码、文本、数据文件）
                # - binary: 编译/打包后的可执行形式
                # - saas: 通过网络服务提供（如API、在线应用）

            result: 
                # 判断usages和forms组合产生的新作品的result
                # 合法值：duplicate, independent, derivative, modification, translation, NODEF
                # 只依据本许可证原文选择，不得套用其他许可或家族的结果：
                # 1. 如果许可证文本明确定义此activity的结果 → 使用文本定义
                #   -duplicate：仅限完全未改变原作品的行为。原文仍约束该副本时，use/copy 的结果为 duplicate。若原文声明不保留任何权利，不要套用这条
                #   -independent: 表示新作品与原作品在法律上是独立的，新作品不受原许可证的条款约束
                #   -derivative：表示新作品与原作品在法律上无法分离，新作品受原许可证的条款约束。例如新作品是原作品的衍生作品，即基于原作品进行了融合、改编等，且需遵守原许可证的条款。重点是原文要求新作品受原许可证约束，不特指“衍生作品”这一个词
                #   -modification：原文把产物描述为对内容的改动，且该改动仍触发与衍生作品相同的义务。仅当原文支持这一结果时使用
                #   -translation：原文把产物描述为语言或格式上的转换。仅当原文支持这一结果时使用
                # 2. 原文讨论了该活动，但没有说明产物是否受本许可约束、能否另行许可 → result 用 NODEF，不要猜成 derivative 或 independent。原文完全没有讨论的活动不要写条目。原文禁止或限制某活动，属于已经讨论，不能因为它是禁止就省略。

            restrictions: []
                # 适用的分发要求
                # 这里的restrictions 是针对特定 usage-result 组合的要求
                # 合法值（可多选，但不得出现未列举的值）：include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction
                # include_use_restriction、include_runtime_restriction 仅当原文对该活动的结果明确要求保留使用限制或运行时限制时列入
                # 当result为 NODEF 时不需要列出此条

            keywords: []
                # 从许可证文本中提取与此活动相关的关键词。每个词都必须在原文中出现，不要引入原文没有的概念。
                # 下面只说明哪类措辞可能对应哪个活动，文本里没有的词不要写入：
                # - modify 一类：modify、adapt、alteration、derivative works
                # - generate 一类：output、result、generation
                # - combine 一类：portion、collection、aggregate
                # - train 一类：train、fine-tune、improve
                # 当result为 NODEF 时不需要列出此条
            relicense: false
                # 定义：针对该 usage和forms组合 产生的结果（Result），用户是否拥有“再许可权”（即：将结果更换为不同于原作品的许可证进行发布）。
                # 注意：有些许可针对同样活动的不同forms（raw/binary/saas）的relicense有不同规定，注意区分
                # 请根据以下逻辑三选一：
                # - true: （允许更换 / 独立作品）
                #     1. 结果被视为“独立作品” (Independent)，不受原协议约束。
                #     2. 原协议允许将此种使用方法下产生的作品闭源或更换协议（只要保留版权声明）。特别注意允许将衍生作品闭源或更换为私有协议，不等于有权力更改原作品本身的许可证。
                # - false: （禁止更换 / 强制继承）
                #     1. 是强 Copyleft 协议的衍生作品：必须严格沿用原协议（传染性）。
                #     2. 禁止分发的协议，或此种使用方法下产生的作品不允许更改协议条款，也不允许转为开源。
                # - conditional: （有条件更换）
                #     允许更换协议，但必须满足特定约束条件 （这里的条件是指存在需要下游遵守的限制性条款，常规的include_notice, include_license, state_changes之类的不属于限制性条款）：
                #     1. 原文允许更换协议，但要求保留特定使用限制。
                #     2. 或者原文要求只能换成其明确列出的兼容许可证。
                # 同一活动在 raw、binary、saas 上的 result 和 relicense 可以不同，但必须各自能在原文中找到依据，不得按许可家族套用固定结论。
                # 当result为 NODEF 时不需要列出此条

            copyleft: false
                # 这里的copyleft 指的是该项特定活动产生的结果是否会触发原许可证的「同许可 / share-alike」传染义务
                # 布尔值：
                # - true: 仅当原文对该活动的可分发结果明确要求以本许可证、或原文点名的兼容许可再发布。使用限制、只要求附上许可文本、或只限制再许可，都不是 copyleft
                # - false: 其余情况
                # 元数据 categories 里如果漏写了 copyleft，不要因此把原文写明的同许可义务改成 false。这一步仍然不要修改元数据本身。
                # 当result为 NODEF 时不需要列出此条
            

                输出格式如下。尖括号只说明该位置填什么；输出时换成从原文得出的枚举值，不要留下尖括号，也不要照搬任何一份已知许可证的结论。元数据各字段保持输入原样。
                  full_name: <官方全称>
                  short_id: <与输入名称完全一致>
                  url: <原文中的官方地址，没有则 null>
                  version: <版本号或 null>
                  available: true
                  categories: [<适用类别>]
                  labels: [<适用标签，没有则 []>]
                  rights: [<原文明确授予的权利>]
                  reserved_rights: [<原文明确保留或禁止的权利>]
                  rights_prefix: [<原文写明的前缀>]
                  coverage: [<原文允许再分发的作品类型>]
                  redistribute: [<原文要求的再分发义务>]
                  compat: []
                  incompat: []
                  terms:
                    - usages: [<原文涉及的活动>]
                      forms: [<原文区分到的形态>]
                      result: <duplicate 或 independent 或 derivative 或 modification 或 translation 或 NODEF>
                      restrictions: [<该活动再分发时原文要求的义务>]
                      keywords: [<该活动在原文中出现的词>]
                      relicense: <true 或 false 或 conditional>
                      copyleft: <true 或 false>

                【最终校验清单】
                    在输出前，请逐项确认：
                    1. terms 中 result 必须与 usage 语义匹配，并且能指回原文。
                    2. 输出内容必须是完整的，包括**许可证元数据**和**terms**
                    3. 所有字段值必须来自预定义枚举。
                    4. 原文没有涉及的活动没有被补写进去。
                """
            messages = [
                {"role": "system", "content": "你是一位专门分析软件和数据许可证的专家。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self._legacy_chat(
                messages, stage="terms", kind="generate", max_tokens=20000
            )
            if not llm_response:
                return None
            self.save_llm_output(license_name, llm_response, "license_analysis_terms")
            yaml_content = re.sub(r"^```(yaml)?\s*|\s*```$", "", llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r"---\s*([\s\S]*?)\s*---", yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            return yaml.safe_load(yaml_content)
        except Exception as e:
            logging.error("Failed to analyze license terms with LLM: %s", e)
            return None

    def evaluate_metadata_compliance(self, license_name: str, license_text: str, metadata: Dict) -> Optional[Dict]:
        """
        阶段一审计：仅对基础元数据进行规则与语义一致性评估
        """
        if not self.api_key:
            logging.error("API key not provided.")
            return None

        try:
            metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)

            prompt = f"""
            你是一位严谨的许可证结构化数据专家。
            你的任务是基于《合法枚举值字典》和《建模规则》，对输入的“基础元数据 YAML”进行编译级查错和语义校验。

            许可证名称: {license_name}
            许可证原文:
            {license_text}

            待复核的基础元数据 YAML:
            {metadata_yaml}

            【合法枚举值字典】
            categories 允许的值:
            - public #只要该许可证是向不特定公众公开发布的（即 Open Source 或 Shared Source），许可证允许社会大众使用，而非针对特定商业伙伴的定制协议，均添加此项          
            - permissive #宽松许可（如 MIT、Apache-2.0），根据实际内容判断，不得通过许可证名称简单臆测            
            - copyleft #copyleft许可（如 GPL、CC-BY-SA）。如果修改或基于某个作品创作衍生作品，并分发该衍生作品，必须使用相同或兼容的许可证来授权整个衍生作品。            
            - disclose #意味着许可作品要提供源代码（如 GPL 要求提供源代码，AFL-3.0要求原始许可人提供源码）            
            - auto-relicensing #具有自动再许可机制（例如允许在特定条件下将衍生作品按兼容许可证再许可）（如 CC-BY-SA、GPL)            
            - software #此许可适用于software           
            - data #此许可适用于data            
            - model #此许可适用于model
            - proprietary # "proprietary" 与 "permissive"（宽松）互斥，不能同时存在; "public" 和 "proprietary" 可以同时出现（例如公开发布文本/源码的专有许可证）   
            # 这里的"适用于"是指此许可本身设计时考虑了software, data, model中的某种或多种，不用考虑现实是否有人用作它用，比如经常有人使用Apache-2.0做model许可，但其实它在设计时并没有考虑model          


            rights 允许的值: # 仅列出许可证明确授予的权利（这里的权利是指不在分发状态下的权利。例如某许可证允许分发未修改的原件、但不允许分发改编版，则 rights 保留 redistribute 与 modify，coverage 只列 duplicate；不要靠删掉 redistribute 来表示禁止改编）仅使用下述值。
              - use
              - modify
              - merge     （"merge" 指将原作品与其他材料组合成集合。仅当原文授予组合、聚合或同类含义的权利时才应出现）
              - redistribute
              - sublicense   
              - commercial_use
              - patent_use
              - copyright  （注意：只有许可证声明放弃版权时，rights 才可能有 copyright）
              # patent_use（专利使用权）指被许可人使用该专利技术的权利。仅当原文明确授予时才应出现在 rights；未提及则不应出现
              # "sublicense" 是指被许可人（Licensee）拥有将自己获得的权利，以自己的名义再次授权给第三方（Sublicensee）的法律权力。只有当许可证文本中明确出现 "sublicense"、"sub-license" 或 "right to grant sublicenses" 等词汇时，才能将此项归入 rights
              # 再分发（redistribute）和再许可（sublicense）不同。redistribute 是指把原作品（及原协议）传给别人；sublicense 是指被许可人有权针对原作品签发一份新的许可协议。redistribute 仅当原文明确允许再分发时才应出现


            reserved_rights 允许的值:
              - trademark_use #（商标使用权）指是否允许被许可人使用该商标。如果许可证不授予商标使用权在reserved_rights列出trademark_use
              - copyright
              - patent  # patent（专利权本身）指的是对某项发明所拥有的完整的专利所有权
              - trademark  #（商标权本身）
              - sublicense #"sublicense" 是指被许可人（Licensee）拥有将自己获得的权利，以自己的名义再次授权给第三方（Sublicensee）的法律权力。只有当许可证文本中明确不允许才能将此项归入reserved_rights
              - commercial_use
              - patent_use  #patent_use（专利使用权）指的是被许可人使用该专利技术的权利
              - redistribute
              # 仅当原文明确保留、拒绝授予或禁止某项权利时才应出现。未提及的权利不要因为许可家族而要求补写
              # 只有当原文明确不授予 patent_use（专利使用权）时，reserved_rights 中才应有 patent_use；未提及则不应有

            coverage 允许的值: #表示哪些类型的作品可以被合法再分发
              - duplicate
              - derivative
              - modification #原文把产物描述为对内容的改动，且该改动仍触发与衍生作品相同的义务。仅当原文支持时才应出现
              - translation #原文把产物描述为语言或格式上的转换。仅当原文支持时才应出现，不要因为许可名称要求补上
              #若原文声明不保留任何权利，所有结果同等对待、没有任何再分发限制，则 coverage 列表应置空
            redistribute 允许的值: #只应包含原文明确要求的项。允许再分发不等于五项都要有。检查多余和遗漏都按原文
              - include_notice #保留版权声明、归因、免责声明，或指向本许可的说明
              - include_license #随附本许可全文，或要求标明本许可并给出许可文本或 URI
              - state_changes #必须说明作品是否被修改。原文没有此要求时不应出现
              - include_use_restriction #再分发时必须保留原文列出的使用限制
              - include_runtime_restriction #再分发或提供服务时必须保留运行时限制

            rights_prefix 允许的值: # 描述所授予权利的性质前缀，仅使用上述值。此部分严格遵循许可证原文，未明确提及的不要臆测。
              - perpetual  #指授权一旦授予，在满足许可证条款的前提下永久有效，不会因时间到期而自动终止, 除非有明文写出（irrevocable不等同于perpetual），否则不得添加
              - worldwide
              - non-exclusive
              - no-charge #排除任何形式的初始许可费
              - royalty-free
              - irrevocable
              - non-transferable  #被授权方不能将所获权利转授给第三方
              - non-sublicensable
              - revocable
              - sublicensable
              - any-person  #文本中直接出现类似 "Permission is hereby granted...to **any person**..."（如 MIT）

            【校验规则】
            1. 枚举必须合法。
            2. 互斥性：rights 与 reserved_rights 必须完全无交集。rights = 被明确授予的使用权，reserved_rights = 明确未授予或禁止的权利，二者应无交集（disjoint）。这里的无交集仅指**完全相同的词汇字符串**不得同时出现在两边（例如 commercial_use 不能同时在两边出现），无需过度引申。
            3. 根据许可证内容检查每一项的值是否存在遗漏
            4. 联动：若 coverage 无 derivative/modification，则 redistribute 禁含 state_changes。
            5. 不得按许可证名称或家族补写或删除 rights、reserved_rights、coverage、compat、incompat。sublicense 仅当原文明确授权时出现在 rights，仅当原文明确拒绝时出现在 reserved_rights。

            【输出要求】（必须严格输出此 YAML，不含 Markdown 代码块）
            chain_of_thought: |
              步骤1：提取 YAML中的列表，逐一对比上方【合法枚举值字典】，确认是否都在字典内。
              步骤2：互斥性检查
              - rights 集合: [此处再次默写 rights]
              - reserved_rights 集合:[此处再次默写 reserved_rights]
              - 真实交集元素: [如有交集写出元素，如无写“无”]
              步骤3：结合文本分析语义是否进行了臆测, 是否有遗漏，是否有错误。注意：权利、保留权利和兼容名单只承认原文明确写出的内容，许可名称上的暗示不算依据；【合法枚举值字典】的拼写和互斥性必须零容忍。
              【针对易错点的特殊防御规则】（审查时必须绝对遵守，不得违背）：
                1. 商业使用一票否决权：只要许可证明确允许商业使用（无论是否有规模限制、行业限制、或者收费要求），`rights` 中必须包含 `commercial_use`，绝对不允许因为存在限制条款就判定 `commercial_use` 为错误。
                2. 再许可的严格定义：在检查此条时必须先在许可证原文中尝试搜寻sublicense，允许用户修改代码并以新协议发布衍生品，绝不等于拥有 `sublicense`（再许可）权！只要协议中明确出现了 "non-transferable"（不可转让）或没有出现明文的 "sublicense"，`sublicense` 权利就未被授予
                3. 再分发义务按原文拆开，不要并成一项，也不要因为允许再分发就要求五项齐备。保留版权声明、归因或免责声明是 include_notice；随附许可全文或要求给出许可文本、URI 是 include_license；原文两者都要求时两项都要有。必须说明做过修改才是 state_changes。绝对禁止建议添加自定义字段。
                4. patent_use 只看原文。原文明确授予才写入 rights；原文明确拒绝才写入 reserved_rights；未提及则两边都不写。不得因作品类型不同而默认补写或默认删除。
                5. redistribute、modify、merge 只看原文是否授予。不得按名称中的限制后缀补写或删除。
                6. compat 与 incompat 只在原文点名兼容或不兼容的许可证时填写。不得按许可家族补写名单。
                7. 一个许可证不可能既是"proprietary"（专有）又是 "permissive"（宽松）的
                8. merge 只在原文授予组合、聚合或同类含义时才应出现
                9. 若原文有专利条款，逐句核对 patent 与 patent_use，不要漏读，也不要在没有该条款时补写

            # 重要格式警告：
            # 1. 所有的 description 和 suggestion 字段的值，必须使用双引号 "" 完全包裹！
            # 2. 在自然语言描述中，绝对禁止使用英文冒号（:），请统一使用中文冒号（：）。
            ok: true/false
            usable_for_compliance: true/false
            rule_conformant: true/false
            semantic_conformant: true/false
            issues: 
              - type: rule | semantic
                field: 字段路径
                severity: critical | warning
                description: "用双引号包裹的中文简述问题"
                suggestion: "用双引号包裹的具体修改建议"
            summary: "中文整体结论"
            """

            messages = [
                {"role": "system",
                 "content": "你是一位严谨的许可证结构化数据专家"},
                {"role": "user", "content": prompt}
            ]

            llm_response = self._legacy_chat(
                messages, stage="metadata", kind="audit", max_tokens=25000
            )
            if not llm_response:
                return None

            # 保存评估报告
            self.save_llm_output(license_name, llm_response, "metadata_evaluation")

            # 解析为 YAML
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            evaluation_report = yaml.safe_load(yaml_content)
            if not isinstance(evaluation_report, dict):
                logging.error("Metadata evaluation report is not a dict")
                return None

            return evaluation_report

        except Exception as e:
            logging.error(f"Failed to evaluate metadata with LLM: {e}")
            return None

    def evaluate_terms_compliance(self, license_name: str, license_text: str, metadata_base: Dict, terms: List[Dict]) -> Optional[Dict]:
        """
        阶段二审计：在基础元数据通过的前提下，仅对 terms / ModelGo 映射部分进行规则与一致性评估。

        返回的评估报告为字典（从 YAML 解析而来），包含字段：
        - ok: bool，本阶段是否通过
        - usable_for_compliance: bool，terms 是否可用于合规性分析
        - rule_conformant: bool，是否符合 terms 建模规则
        - semantic_conformant: bool，是否与许可证原文及基础元数据含义一致
        - issues: 列表，逐条给出问题及修改建议（field 仅限 terms[...] 路径）
        - summary: 对整体情况的中文总结
        """
        if not self.api_key:
            logging.error("API key not provided. Please set it when initializing LLMLicenseHelper")
            return None

        try:
            metadata_yaml = yaml.dump(metadata_base, allow_unicode=True, sort_keys=False)
            terms_yaml = yaml.dump({"terms": terms}, allow_unicode=True, sort_keys=False)

            prompt = f"""
           你是一位专业的许可证分析与建模专家，精通开源软件、开放数据、AI 模型及内容许可证。
           下面给出：
           1）某许可证的原文，
           2）已经通过基础审计的元数据（不含 terms，**仅作只读参考**），
           3）在该元数据基础上生成的 terms / ModelGo 映射。
           请你**只针对 terms 部分**进行审计；不得对元数据字段开 issue，也不建议回退修改元数据。

           许可证名称:
           {license_name}

           许可证原文:
           {license_text}

           基础元数据 YAML（已通过基础审计，不含 terms）:
           {metadata_yaml}

           待审计的 terms YAML:
           {terms_yaml}
            
           # 以下描述中的原作品是指使用当前分析许可证的data/software/model

           一、规则维度（terms 建模规则）：
           1. usages 只能使用以下活动名：
              [use, copy, modify, augment, train, combine, combine_mix, amalgamate, distill, generate, embed, stat]
               #forms,result,restrictions,relicense,copylef都相同的usage合并写在一起是合理的，但是除上述枚举活动不得擅自增加不存在的活动
               #每个usages和forms的组合只能出现一次
               #原文已经讨论的活动（包括允许、禁止、有限制）必须列举。原文已经写明产物法律后果时，result 不得为 NODEF；讨论了但没写明产物地位时可以用 NODEF，不要猜。原文完全没有提及的活动不要要求补写，也不要因为许可证名称或家族要求补上训练、生成等活动。
               #例如原文不允许利用输出改进某种模型，才是对 distill 的限制。不得仅因本许可用于模型、或原文出现“衍生作品”，就要求写入 train。
               #copy  #新作品是原作品的复制品
               #use  #单纯的使用，这里不产生新作品，后续分析时新作品等同原作品
               #modify #新作品modify了原作品
               #train #新作品基于原作品训练而成，例如新作品由原作品微调而成
               #combine #将多部作品(同种类型作品的组合，例如数据和数据的组合, 原作品是其中之一，没有对原作品进行修改, 只是组合）组合起来构建一部新作品。或者将原作品放在另外一个大的作品集合中（同类型作品）。组合后的作品整体是新作品。仅当原文涉及组合、聚合或同类安排时才应出现
               #combine_mix #将多个作品组合以构建一个新作品（原作品是其中之一），由此产生的新作品是一种混合类型的组合（可能包括data, model, software多种类型作品进行组合, 没有对原作品进行修改，只是组合）。或者将原作品放在另外一个大的作品集合中（混合类型作品）。组合后的作品整体是新作品。
               #amalgamate #将多个作品进行融合（原作品是其中之一），生成无法再分离的新作品。或者将原作品融入另一个作品中。融合后的作品整体是新作品。
               #augment #在不改变数据核心标签含义的前提下，对原始数据进行一系列随机或预设的变换，从而得到更多、更多样化的新样本，新作品是原作品（通常是data）增强产生的
               #distill # 模型蒸馏，Distill knowledge from old models to new models。新作品对原作品进行了蒸馏
               #generate #原作品生成新作品。仅当原文明确规范输出、生成结果或同类产物时才应出现。不要因为许可名称或其他家族的惯例要求写入 generate
               #embed # Embed works (corpus, image or other data samples) using aux_works (model or algorithm)，将原作品（通常是data）利用 model 或者 algorithm 嵌入到新作品
               #stat #对数据进行统计，新作品是原作品的统计分析数据
               
           2. forms 只能为 [raw, binary, saas] 中的一个或多个。
           3. result 只能为 [duplicate, independent, derivative, modification, translation, NODEF]。
              # 判断usages和forms组合产生的新作品的result。只依据本许可证原文，不得套用其他许可或家族的结果
              - duplicate：仅限完全未改变原作品的行为。原文仍约束该副本时，use/copy 的结果为 duplicate。若原文声明不保留任何权利，不要套用这条
              - independent: 表示新作品与原作品在法律上是独立的，新作品不受原许可证的条款约束
              - derivative：表示新作品与原作品在法律上无法分离，新作品受原许可证的条款约束。例如新作品是原作品的衍生作品，即基于原作品进行了融合、改编等，且需遵守原许可证的条款。重点是原文要求新作品受原许可证约束，不特指“衍生作品”这一个词
              - modification：强调对原作品进行了内容层面的改动，同样触发衍生作品义务。仅当原文支持这一结果时使用
              - translation：特指对原作品进行语言或格式上的转换（如文本翻译、代码转写）。仅当原文支持这一结果时使用
              - NODEF：原文讨论了该活动，但没有说明产物是否受本许可约束、能否另行许可。此时不要要求改成 derivative 或 independent。原文完全没讨论的活动不应出现在 terms 里
              #注意：如果原文中找不到任何支持某种活动的条款，审计不得要求补写该活动，也不得仅因「可理解为分发」强制并入 copy
              #使用限制不是 copyleft。输出不受本许可约束，与使用输出时仍须遵守使用限制，要分开看。原文把可分离的并列和融成一部作品分开规定时，不要要求合并成同一个 result；原文没有这种区分时，不要要求拆开。           
           4. restrictions 只能为 [include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]。# 这里的restrictions 是针对特定 usage-result 组合分发时的要求 # 当result为 NODEF 时不需要列出此条
           5. relicense 只能为 [true, false, conditional]。
              # 定义：针对该 usage和forms组合 产生的结果（Result），用户是否拥有“再许可权”（即：将结果更换为不同于原作品的许可证进行发布）。
              注意：有些许可针对同样活动的不同forms（raw/binary/saas）的relicense有不同规定，注意区分
               # 根据以下逻辑三选一：
               # - true: （允许更换 / 独立作品）
               #     1. 结果被视为“独立作品” (Independent)，不受原协议约束。
               #     2. 原协议允许将此种使用方法下产生的作品闭源或更换协议（只要保留版权声明）。特别注意允许将衍生作品闭源或更换为私有协议，不等于有权力更改原作品本身的许可证。允许 sublicense 也不等于可以更换未产生新作品的副本的许可证，应结合原文判断
               # - false: （禁止更换 / 强制继承）
               #     1. 是强 Copyleft 协议，协议有要求此活动产生的作品必须严格沿用原协议（传染性）（就算使用兼容许可也不允许，必须沿用原许可的情况）。
               #     2. 禁止分发的协议，或此种使用方法下产生的作品不允许更改协议条款，也不允许转为开源。
               #     3. 如果 result 为 duplicate，且原文仍约束该副本，relicense 应为 false。若原文声明不保留任何权利，不要套用这条
               # - conditional: （有条件更换）
               #     允许更换协议，但必须满足特定约束条件 （这里的条件是指存在需要下游遵守的限制性条款，常规的include_notice, include_license, state_changes之类的不属于限制性条款）：
               #     1. 原文允许更换协议，但要求保留特定使用限制。
               #     2. 或者原文要求只能换成其明确列出的兼容许可证。
               # 同一活动在不同 forms 上的 result 和 relicense 可以不同，但必须各自能在原文中找到依据。不得按许可家族要求固定的 relicense。
               # 当result为 NODEF 时不需要列出此条
           6. copyleft 只能为布尔值 true/false。
                # 这里的copyleft 指的是该项特定活动产生的结果是否会触发原许可证的「同许可 / share-alike」传染义务
                # 布尔值：
                # - true: 仅当原文对该活动的可分发结果明确要求以本许可证、或原文点名的兼容许可再发布。使用限制、只要求附上许可文本、或只限制再许可，都不是 copyleft
                # - false: 其余情况
                # 元数据 categories 漏写 copyleft 时，不要因此要求把原文写明的同许可义务改成 false。审计仍不得建议修改元数据。
                # 当result为 NODEF 时不需要列出此条
           

           二、与许可证原文的语义关系： 若许可证文本清晰规定了某类行为（例如衍生作品、输出、训练等）的法律后果，应优先按照文本定义检视 result/restrictions/relicense。
           

           三、审计范围（必须遵守）：
           1. 基础元数据 YAML 仅作只读上下文，**不得**对 rights、redistribute、coverage 等 metadata 字段开 issue。不得用元数据里的 compat 按许可家族反推 terms 的 relicense。
           2. issues 的 field 路径**必须**为 terms[...] 形式（例如 "terms[0].restrictions"、"terms[1].result"）。
           3. 分发时需附带协议、署名、使用限制等问题，应通过修改对应 terms 条目的 restrictions / result / relicense 表达，**不得**建议修改元数据或回退到元数据层。
           4. 若 metadata.redistribute 与某 usage 的分发义务相关，suggestion 应指向具体 terms[n].restrictions，而非 metadata 字段。
           5. 审计范围只限定于现有设计，如果你发现存在当前枚举值无法表达清楚的内容，仅作提醒，但不可以虚构枚举值内没有的内容，也不可以修改为枚举值内没有的内容，不得以此作为ok: false的依据
           6. 如果你发现某活动被许可证禁止，terms里存在相关活动，这是对的，被建模不代表该许可允许某个活动，而是涉及某个活动，不论是允许还是禁止
           
           你的任务：
           1. 检查 terms 是否违反上述建模规则。
           2. 检查 terms 是否在语义上明显偏离许可证文本。
           3. 对每个问题给出分类和修改建议（给出建议时不得擅自建议添加规则内没有提及的词汇，也不应该建议改动建模规则，给出的建议要在规则之内；suggestion 必须可在 terms 层直接落实）：

           输出要求（必须严格遵守）：
           - 只输出一段 YAML，不要包含任何解释性自然语言，不要使用 Markdown 代码块（如 ```yaml）。
           # 格式警告：所有 description 和 suggestion 必须用双引号 "" 包裹！绝对禁止在自然语言中使用英文冒号（:），请统一使用中文冒号（：）！
           - 输出 YAML 结构必须如下（可以根据需要在 issues 中增加多条记录）：
             chain_of_thought: |
             你必须按以下步骤：
             - 步骤1：建模规则检查
             - 步骤2：检查语义上是否明显偏离许可证文本
             ok: true/false
             usable_for_compliance: true/false  #terms 是否可用于合规性分析
             rule_conformant: true/false #是否符合 terms 建模规则
             semantic_conformant: true/false #是否与许可证原文含义一致
             issues:
               - type: rule | semantic | mapping
                 field: 字段路径（必须为 "terms[0].usages"、"terms[1].result" 等 terms 内路径）
                 severity: critical | warning
                 description: "用双引号包裹的中文问题描述"
                 suggestion: "用双引号包裹的中文修改建议（必须可在 terms 层直接落实）"
             summary: "中文整体总结"
           - ok 为 true 仅当 usable_for_compliance 为 true 且 issues 中不存在 severity 为 critical 的条目；仅有 warning 时 ok 可为 true。
           """

            messages = [
                {"role": "system", "content": "你是一位严谨的许可证 terms / ModelGo 映射审计专家"},
                {"role": "user", "content": prompt},
            ]

            llm_response = self._legacy_chat(
                messages, stage="terms", kind="audit", max_tokens=25000
            )
            if not llm_response:
                return None

            # 保存评估报告
            self.save_llm_output(license_name, llm_response, "terms_evaluation")

            # 解析为 YAML
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            evaluation_report = yaml.safe_load(yaml_content)
            if not isinstance(evaluation_report, dict):
                logging.error("Terms evaluation report is not a dict")
                return None
            return evaluation_report

        except Exception as e:
            logging.error(f"Failed to evaluate terms with LLM: {e}")
            return None

    def _metadata_audit_passed(self, report: Optional[Dict]) -> bool:
        """阶段一审计是否通过：ok 且 usable_for_compliance 为 True，且无 critical 问题。"""
        if not report:
            return False
        if report.get("ok") is not True or report.get("usable_for_compliance") is not True:
            return False
        issues = report.get("issues") or []
        return not any(
            (isinstance(i, dict) and i.get("severity") == "critical") for i in issues
        )

    def _terms_audit_passed(self, report: Optional[Dict]) -> bool:
        """阶段二审计是否通过：ok 且 usable_for_compliance 为 True，且无 critical 问题。"""
        if not report:
            return False
        if report.get("ok") is not True or report.get("usable_for_compliance") is not True:
            return False
        issues = report.get("issues") or []
        return not any(
            (isinstance(i, dict) and i.get("severity") == "critical") for i in issues
        )

    def _evaluate_audit_with_retry(
        self,
        license_name: str,
        evaluate_fn,
        *,
        stage_label: str,
        stage_key: str = "",
        passed_fn=None,
    ) -> Optional[Dict]:
        """
        调用审计 evaluate；若返回 None（API 断连、YAML 解析失败等），
        在 MAX_AUDIT_EVAL_RETRY 内重试，**不计入 fix 轮次**。
        """
        report: Optional[Dict] = None
        for eval_attempt in range(MAX_AUDIT_EVAL_RETRY):
            report = evaluate_fn()
            if report is not None:
                if stage_key:
                    passed = bool(passed_fn(report)) if passed_fn else True
                    self._tag_last_legacy(stage_key, "audit", audit_passed=passed)
                if eval_attempt > 0:
                    logging.info(
                        "%s 审计在第 %d 次 evaluate 重试后获得有效报告（许可证: %s）",
                        stage_label,
                        eval_attempt + 1,
                        license_name,
                    )
                return report
            if stage_key:
                self._tag_last_legacy(stage_key, "audit", audit_passed=False)
            if eval_attempt < MAX_AUDIT_EVAL_RETRY - 1:
                wait = LLM_CALL_RETRY_BACKOFF_S * (2 ** eval_attempt)
                logging.warning(
                    "%s 审计返回 None，%.0fs 后进行第 %d 次 evaluate 重试（不计 fix 轮次）（许可证: %s）",
                    stage_label,
                    wait,
                    eval_attempt + 2,
                    license_name,
                )
                time.sleep(wait)
        logging.error(
            "%s 审计在 %d 次 evaluate 后仍为 None（许可证: %s）",
            stage_label,
            MAX_AUDIT_EVAL_RETRY,
            license_name,
        )
        return None

    def fix_metadata_with_llm(
        self,
        license_name: str,
        license_text: str,
        metadata: Dict,
        report: Dict,
        attempt: int = 1,
    ) -> Optional[Dict]:
        """
        根据阶段一审计报告，调用 LLM 修正基础元数据（不含 terms）。
        返回修正后的 metadata 字典，失败返回 None。修正后的内容会保存到 llm_outputs/revisions/ 供检查。
        """
        if not self.api_key:
            return None
        try:
            metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
            report_yaml = yaml.dump(report, allow_unicode=True, sort_keys=False)

            prompt = f"""
            你是一位严谨的许可证结构化数据专家。请根据「审计报告」对「当前基础元数据」做**最小必要修正**，输出**且仅输出**修正后的完整基础元数据 YAML（不包含 terms 字段）。

            【硬性约束 — 输出必须满足】
            1. short_id 必须与许可证名称完全一致：{license_name}
            2. 不得输出 terms 字段；字段顺序保持与当前元数据一致
            3. categories 仅允许：[public, software, data, model, proprietary, permissive, copyleft, disclose, auto-relicensing]
               - proprietary 与 permissive 互斥；public 与 proprietary 可同时存在
            4. rights 仅允许：[use, modify, merge, redistribute, sublicense, commercial_use, patent_use, copyright]
               - patent_use 仅当原文明确授予时才可列入 rights
               - sublicense 仅当原文明确授权时才可列入 rights
            5. reserved_rights 仅允许：[trademark_use, copyright, patent, trademark, sublicense, commercial_use, patent_use, redistribute]
               - 不得含 modify、copy、share 等行为动词
               - patent_use 仅当原文明确拒绝授予时才可列入 reserved_rights；未提及则不要补写
            6. rights 与 reserved_rights 同一字符串不得同时出现（完全无交集）
            7. rights_prefix 仅允许：[perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable, non-transferable, non-sublicensable, revocable, sublicensable, any-person]
            8. coverage 仅允许：[duplicate, derivative, modification, translation]；原文声明不保留任何权利时可以置空
            9. redistribute 仅允许：[include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]
               - 若 coverage 不含 derivative/modification，则 redistribute 不得含 state_changes
            10. compat / incompat 为许可证 ID 列表；只保留原文点名的许可证，不得按家族补写，不得自造字段或枚举外取值

            【修正原则】
            1. 严格依据审计报告 issues 逐条修改；仅修改 issues 中 field 指向的字段，未提及字段保持原值
            2. 若 suggestion 给出具体列表赋值（如 redistribute 添加 include_notice），必须原样落实，不得遗漏
            3. 若多条 suggestion 冲突，优先满足 severity 为 critical 的项，且以枚举合规为准
            4. 不得添加审计未要求的自定义字段；不得臆测原文未提及的 rights_prefix
            5. 不要输出任何解释、Markdown 代码块或 chain_of_thought，只输出 YAML

            许可证名称: {license_name}

            许可证原文（语义对照）:
            {license_text}

            审计报告:
            {report_yaml}

            当前基础元数据（待修正）:
            {metadata_yaml}

            请直接输出修正后的完整基础元数据 YAML（字段顺序保持与当前元数据一致，不包含 terms）。
            """

            messages = [
                {"role": "system", "content": "你输出且仅输出合规的 YAML，无解释、无 Markdown 标记。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self._legacy_chat(
                messages, stage="metadata", kind="fix", max_tokens=4000
            )
            if not llm_response:
                return None

            self.save_llm_output(license_name, llm_response, "license_metadata_fix")
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            fixed = yaml.safe_load(yaml_content)
            if isinstance(fixed, dict) and "terms" in fixed:
                del fixed["terms"]
            if fixed:
                self.save_revision_for_inspection(
                    license_name, fixed, "metadata_fix", attempt
                )
            return fixed
        except Exception as e:
            logging.error(f"fix_metadata_with_llm failed: {e}")
            return None

    def fix_terms_with_llm(
        self,
        license_name: str,
        license_text: str,
        metadata_base: Dict,
        terms: List[Dict],
        report: Dict,
        attempt: int = 1,
    ) -> Optional[List[Dict]]:
        """
        根据阶段二审计报告，调用 LLM 修正 terms（仅修改 terms 列表，不修改元数据）。
        返回修正后的 terms 列表，失败返回 None。修正后的内容会保存到 llm_outputs/revisions/ 供检查。
        """
        if not self.api_key:
            return None
        try:
            terms_yaml = yaml.dump({"terms": terms}, allow_unicode=True, sort_keys=False)
            report_yaml = yaml.dump(report, allow_unicode=True, sort_keys=False)
            metadata_yaml = yaml.dump(metadata_base, allow_unicode=True, sort_keys=False)

            prompt = f"""
            你是一位专业的许可证 terms / ModelGo 映射专家。请根据「审计报告」对「当前 terms」做**最小必要修正**，输出**且仅输出**修正后的 terms YAML（格式为 terms: [...]）。

            【硬性约束 — 输出必须满足】
            1. usages 仅允许：[use, copy, modify, augment, train, combine, combine_mix, amalgamate, distill, generate, embed, stat]
            2. forms 仅允许：[raw, binary, saas]
            3. result 仅允许：[duplicate, independent, derivative, modification, translation, NODEF]
            4. restrictions 仅允许：[include_notice, include_license, state_changes, include_use_restriction, include_runtime_restriction]
            5. relicense 仅允许：[true, false, conditional]；copyleft 为布尔值 true/false
            6. 原文已经写明产物法律后果时，result 不得为 NODEF。原文讨论了该活动但没有说明产物地位时，保留 NODEF，不要改猜成 derivative 或 independent
            7. 禁止自造 usage、forms、result 等枚举外取值（例如 none、source、distribute、reverse_engineer 等均非法）
            8. 原文禁止某活动时仍保留该活动。result 按原文对产物的规定填写；原文只禁止、未规定产物地位时用 NODEF，不要默认写成 derivative。需要保留的使用限制写入 restrictions，不得用自造活动名表达禁止
            9. 当 result 为 NODEF 时，可不列出 restrictions、relicense、copyleft、keywords
            10. 每个 usages 与 forms 的组合在 terms 中只能出现一次
    

            【修正原则】
            1. 严格依据审计报告 issues 逐条修改；仅修改 issues 中 field 路径指向的字段，未提及的 term 条目及其字段保持原值
            2. 若 suggestion 给出具体赋值（如 restrictions 设为 [include_license, include_notice]），必须原样落实，不得输出空 restrictions: []
            3. field 以 terms[ 开头的问题只在 terms 内修正，不得改动已锁定的基础元数据
            4. 若多条 suggestion 冲突，优先满足 severity 为 critical 的项，且以枚举合规为准
            5. 不得删除现有合法 term 条目，除非审计明确要求合并或删除
            6. 如果审计要求你在原有某条目的基础上修改，你应该在要求的原位置修改，不得导致出现一个“usages 与 forms 的组合”在 terms 中出现多次
            7. 不要输出任何解释、Markdown 代码块或 chain_of_thought，只输出 YAML

            许可证名称: {license_name}

            许可证原文（语义对照）:
            {license_text}

            已锁定的基础元数据:
            {metadata_yaml}

            审计报告:
            {report_yaml}

            当前 terms（待修正）:
            {terms_yaml}

            请直接输出修正后的 terms YAML，格式为 terms: 后跟列表。
            """

            messages = [
                {"role": "system", "content": "你输出且仅输出合规的 YAML，无解释、无 Markdown 标记。"},
                {"role": "user", "content": prompt},
            ]
            llm_response = self._legacy_chat(
                messages, stage="terms", kind="fix", max_tokens=8000
            )
            if not llm_response:
                return None

            self.save_llm_output(license_name, llm_response, "license_terms_fix")
            yaml_content = re.sub(r'^```(yaml)?\s*|\s*```$', '', llm_response.strip(), flags=re.MULTILINE)
            yaml_match = re.search(r'---\s*([\s\S]*?)(?:\s*---|$)', yaml_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            fixed_block = yaml.safe_load(yaml_content)
            if isinstance(fixed_block, dict):
                fixed_terms = fixed_block.get("terms") or []
            else:
                fixed_terms = fixed_block or []
            if fixed_terms:
                self.save_revision_for_inspection(
                    license_name, {"terms": fixed_terms}, "terms_fix", attempt
                )
            return fixed_terms
        except Exception as e:
            logging.error(f"fix_terms_with_llm failed: {e}")
            return None

    def _default_terms(self) -> List[Dict]:
        """返回最小可用 terms 列表，供 get_license_data 在 LLM 未返回 terms 时使用。"""
        return [
            {
                "usages": ["use", "copy", "combine", "generate", "train", "embed", "distill", "amalgamate", "modify"],
                "forms": ["raw", "binary", "saas"],
                "result": "NODEF",
                "restrictions": [],
                "relicense": True,
            }
        ]

    def get_license_data(self, license_name: str) -> Optional[Dict]:
        """
        获取未知许可证的数据。默认走现有两阶段审计与自动修复循环。

        环境变量 LICENSE_MODELING_PIPELINE=qa 时切换到抽取+问卷+装配后端
        （phase 1；失败回退本方法的 legacy 实现）。
        """
        pipeline = resolve_license_modeling_pipeline()
        if pipeline == LICENSE_MODELING_PIPELINE_QA:
            try:
                from scripts.license_modeling_qa.pipeline import run_qa_modeling
            except ImportError:
                from license_modeling_qa.pipeline import run_qa_modeling

            return run_qa_modeling(self, license_name)
        return self.get_license_data_legacy(license_name)

    def get_license_data_legacy(self, license_name: str) -> Optional[Dict]:
        """
        现有两阶段审计与自动修复循环（legacy）：
        1. 生成基础元数据 -> 阶段一审计 -> 不通过则 LLM 修正并重试，直到通过或达到最大重试次数。
        2. 元数据锁定后生成 terms -> 阶段二审计 -> 不通过则仅修正 terms 并重试。
        3. 审计 evaluate 若返回 None（API/YAML 失败），在 fix 循环内单独重试 evaluate，不计 fix 轮次。
        4. 合并并返回最终 license_data。
        """
        # 1. 获取许可证文本
        license_text = self.fetch_license_text(license_name)
        if not license_text:
            logging.info("未获取到许可证全文，将仅基于名称由 LLM 推断元数据: %s", license_name)
            license_text = f"{license_name}"

        # 2. 生成基础元数据
        metadata_base = self.analyze_license_with_llm_data(license_name, license_text)
        if not metadata_base:
            return None

        # 3. 阶段一审计 + 自动修复循环
        metadata_evaluation = None
        for fix_attempt in range(MAX_METADATA_RETRY):
            metadata_evaluation = self._evaluate_audit_with_retry(
                license_name,
                lambda: self.evaluate_metadata_compliance(
                    license_name, license_text, metadata_base
                ),
                stage_label="基础元数据",
                stage_key="metadata",
                passed_fn=self._metadata_audit_passed,
            )
            if self._metadata_audit_passed(metadata_evaluation):
                break
            if fix_attempt < MAX_METADATA_RETRY - 1 and metadata_evaluation:
                logging.info(
                    f"基础元数据未通过阶段一审计，第 {fix_attempt + 1} 次自动修正（许可证: {license_name}）。"
                )
                fixed = self.fix_metadata_with_llm(
                    license_name, license_text, metadata_base, metadata_evaluation,
                    attempt=fix_attempt + 1,
                )
                self._tag_last_legacy("metadata", "fix", fix_applied=bool(fixed))
                if fixed:
                    metadata_base = fixed
                else:
                    logging.warning("LLM 修正基础元数据失败，保留当前版本并继续。")
                    break
            else:
                if not metadata_evaluation:
                    logging.warning(
                        f"基础元数据审计无法获得有效报告（许可证: {license_name}）。"
                    )
                else:
                    logging.warning(
                        f"基础元数据在 {MAX_METADATA_RETRY} 次内未通过阶段一审计，请根据评估报告人工修正（许可证: {license_name}）。"
                    )
                break

        # 4. 元数据锁定，生成包含 terms 的完整数据
        metadata_base_locked = copy.deepcopy(metadata_base)
        self.save_revision_for_inspection(
            license_name, metadata_base_locked, "metadata_locked", 0
        )
        metadata_base_str = yaml.dump(metadata_base_locked, allow_unicode=True, sort_keys=False)
        license_data = self.analyze_license_with_llm_terms(
            license_name, license_text, metadata_base_str
        )
        if not license_data:
            return None

        terms = license_data.get("terms", []) or []

        # 5. 阶段二审计 + 自动修复循环（仅修改 terms）
        if terms:
            terms_evaluation = None
            for fix_attempt in range(MAX_TERMS_RETRY):
                terms_evaluation = self._evaluate_audit_with_retry(
                    license_name,
                    lambda: self.evaluate_terms_compliance(
                        license_name=license_name,
                        license_text=license_text,
                        metadata_base=metadata_base_locked,
                        terms=terms,
                    ),
                    stage_label="terms",
                    stage_key="terms",
                    passed_fn=self._terms_audit_passed,
                )
                if self._terms_audit_passed(terms_evaluation):
                    break
                if fix_attempt < MAX_TERMS_RETRY - 1 and terms_evaluation:
                    logging.info(
                        f"terms 未通过阶段二审计，第 {fix_attempt + 1} 次自动修正（许可证: {license_name}）。"
                    )
                    fixed_terms = self.fix_terms_with_llm(
                        license_name,
                        license_text,
                        metadata_base_locked,
                        terms,
                        terms_evaluation,
                        attempt=fix_attempt + 1,
                    )
                    self._tag_last_legacy("terms", "fix", fix_applied=bool(fixed_terms))
                    if fixed_terms:
                        terms = fixed_terms
                        license_data["terms"] = terms
                    else:
                        logging.warning("LLM 修正 terms 失败，保留当前版本并继续。")
                        break
                else:
                    if not terms_evaluation:
                        logging.warning(
                            f"terms 审计无法获得有效报告（许可证: {license_name}）。"
                        )
                    else:
                        logging.warning(
                            f"terms 在 {MAX_TERMS_RETRY} 次内未通过阶段二审计，请根据评估报告人工修正（许可证: {license_name}）。"
                        )
                    break

        # 6. 验证和完善最终数据
        if "short_id" not in license_data:
            license_data["short_id"] = license_name
        required_fields = ["full_name", "short_id", "url", "categories", "rights", "reserved_rights", "terms"]
        for field in required_fields:
            if field not in license_data:
                logging.warning(f"Missing required field {field} in license data")
                if field == "full_name":
                    license_data[field] = license_name
                elif field == "url":
                    license_data[field] = f"https://example.com/license/{license_name}"
                elif field in ["categories", "rights", "reserved_rights"]:
                    license_data[field] = []
                elif field == "terms":
                    license_data[field] = []

        # 7. 保存最终合并结果供检查
        self.save_revision_for_inspection(
            license_name, license_data, "license_final", 0
        )
        return license_data

    # --------- 未知许可证处理（去掉交互式 input）---------

    def handle_unknown_license(self, component_name: str) -> Tuple[str, Optional[str]]:
        """
        处理未知许可证的组件（无交互版本，适用于服务端场景）。
        尝试顺序：
        1）LLM 按组件名猜测许可证；
        2）GitHub/HF API 查询；
        3）从 README / LICENSE 文本中用 LLM 进一步识别；
        4）都失败则保守返回 "Unlicense"。
        """
        logging.info("组件 %s 许可证未知，将尝试自动查询。", component_name)
        final_name = component_name

        try:
            # 1. 使用 LLM 直接按组件名查询
            license_name = self.query_license_by_component_name(final_name)
            if license_name:
                logging.info("通过 LLM 查询到组件 %s 的许可证：%s", final_name, license_name)
                return license_name, None

            # 2. 尝试通过 API 查询（GitHub / Hugging Face）
            logging.info("LLM 未能直接确定许可证，尝试通过 API 查询...")
            api_result = fetch_license_from_api(final_name)
            if api_result:
                license_name, license_text, readme_text = api_result
                logging.info("API 查询成功：组件 %s 许可证为 %s", final_name, license_name)

                if license_name is None or (isinstance(license_name, str) and license_name.lower() == "other"):
                    logging.info(
                        "API 返回的许可证为 %s，将尝试从 README 或 LICENSE 内容中进一步分析。",
                        license_name,
                    )
                    analyzed_license = None
                    if readme_text:
                        analyzed_license = self.analyze_license_from_readme(final_name, readme_text)
                    if not analyzed_license and license_text:
                        analyzed_license = self.analyze_license_text_for_component(final_name, license_text)
                    if analyzed_license:
                        license_name = analyzed_license
                        logging.info("从内容中分析出许可证名称：%s", license_name)
                    else:
                        license_name = f"{component_name}_license"
                        logging.info("无法从内容中分析出许可证名称，使用占位名：%s", license_name)

                    if license_text or readme_text:
                        self.save_license_raw_content(license_name, license_text or readme_text)

                return license_name, None

            logging.warning(
                "API 查询失败：将设置该组件许可证为 Unlicense 以避免与其他组件产生冲突，结果仅供参考。"
            )
            return "Unlicense", None

        except Exception as e:
            logging.error("处理未知许可证时发生错误: %s", e)
            return "Unlicense", str(e)

    # --------- 其余辅助函数：与 ModelGo2-main 基本一致 ---------
    # query_license_by_component_name / analyze_license_text_for_component /
    # analyze_license_from_readme / save_license_raw_content ...

    def query_license_by_component_name(self, component_name: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请诚实回答：你知道'{component_name}'这个组件的默认许可证是什么吗？
            如果你确实知道，请直接返回许可证名称，不要添加任何其他解释。
            如果你不知道或不确定，请直接回答'不知道'。
            """
            messages = [
                {"role": "system", "content": "你是一位软件和数据许可证专家，只回答你确定的事实。"},
                {"role": "user", "content": prompt},
            ]
            answer = self.call_llm(messages=messages, temperature=0.3, max_tokens=150)
            if not answer:
                return None
            self.save_llm_output(component_name, answer, "component_license_query")
            if answer.lower() in ["不知道", "不", "no", "don't know", "i don't know"]:
                return None
            return answer.strip()
        except Exception as e:
            logging.error("查询组件许可证时发生错误: %s", e)
            return None

    def analyze_license_text_for_component(self, component_name: str, license_text: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请分析以下文本，确定'{component_name}'组件使用的许可证名称。
            
            文本内容：
            {license_text}
            
            请根据文本内容直接返回许可证名称，如果无法确定，请返回'Unknown'。
            """
            messages = [
                {"role": "system", "content": "你是一位许可证分析专家，能够从文本中识别标准许可证。"},
                {"role": "user", "content": prompt},
            ]
            license_name = self.call_llm(messages=messages, temperature=0.1, max_tokens=50)
            if not license_name:
                return None
            self.save_llm_output(component_name, license_name, "license_text_analysis")
            if license_name == "Unknown":
                return None
            return license_name
        except Exception as e:
            logging.error("分析许可证文本时发生错误: %s", e)
            return None

    def analyze_license_from_readme(self, component_name: str, readme_content: str) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            prompt = f"""
            请从以下 README 内容中分析'{component_name}'组件使用的许可证名称。
            
            README 内容：
            {readme_content}
            
            请直接返回许可证名称，如果无法确定，请返回'Unknown'。
            """
            messages = [
                {"role": "system", "content": "你是一位许可证分析专家，能够从 README 中识别标准许可证。"},
                {"role": "user", "content": prompt},
            ]
            license_name = self.call_llm(messages=messages, temperature=0.1, max_tokens=50)
            if not license_name or license_name == "Unknown":
                return None
            return license_name
        except Exception as e:
            logging.error("Failed to analyze license from README for component %s: %s", component_name, e)
            return None

    def save_license_raw_content(self, license_name: str, license_content: str) -> str:
        base_dir = os.path.join(os.path.dirname(__file__), "license_raw")
        os.makedirs(base_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9-_.]", "-", license_name)
        file_path = os.path.join(base_dir, f"{safe_name}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(license_content)
            logging.info("License content saved to: %s", file_path)
            print(f"许可证全文已保存到: {file_path}")
        except Exception as e:
            logging.error("Failed to save license content to %s: %s", file_path, e)
            print(f"保存许可证全文失败: {e}")
        return file_path


# 全局实例
llm_helper = LLMLicenseHelper()


def set_api_key(api_key: str, model: str = "deepseek", github_token: Optional[str] = None) -> LLMLicenseHelper:
    """
    设置全局 llm_helper 的 API key 和模型。
    """
    global llm_helper
    supported_models = build_supported_models()
    preset = normalize_license_llm_preset(model)

    llm_helper.api_key = resolve_license_llm_api_key(model, api_key)

    if preset in supported_models:
        llm_helper.model_type = preset
        llm_helper.model = supported_models[preset]["model"]
        llm_helper.api_base = supported_models[preset]["api_base"]
    else:
        llm_helper.model_type = "custom"
        llm_helper.model = model

    llm_helper.github_token = github_token
    logging.info("已配置 %s API key 用于许可证分析", model)
    return llm_helper

