"""可选的未知许可建模后端（抽取 + 问卷 + 装配）。

默认不启用。设置 LICENSE_MODELING_PIPELINE=qa 后，`get_license_data` 会走本包。
Phase 1：词表预抽 → L0/Combine/modify 问卷 → 装配 YAML；失败回退 legacy。
"""

from .pipeline import run_qa_modeling

__all__ = ["run_qa_modeling"]
