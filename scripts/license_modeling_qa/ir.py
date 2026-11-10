"""问卷填出的法律事实 IR。装配器只读这里，不读 ModelGo 字段名以外的 YAML。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


YES_NO = ("yes", "no", "not_mentioned")
_WORK_TYPES = ("software", "data", "model")


def work_types(value: object) -> list[str]:
    """作品类型。多选列表，或旧的单个 software|data|model。"""
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            tok = str(item).strip().lower()
            if tok in _WORK_TYPES and tok not in out:
                out.append(tok)
        return out
    text = str(value or "").strip().lower()
    if text in _WORK_TYPES:
        return [text]
    return []


def resolved_work_types(current: object, cues: object) -> list[str]:
    """问卷选择与定义句取并。线索只补充类型，抽不中时不删问卷已选的类型。"""
    types = work_types(current)
    if getattr(cues, "model_work", False) and "model" not in types:
        types.append("model")
    if getattr(cues, "content_data_work", False) and "data" not in types:
        types.append("data")
    if getattr(cues, "software_work", False) and "software" not in types:
        types.append("software")
    return types


def store_work_signals(types: list[str]) -> str | list[str]:
    if not types:
        return "not_mentioned"
    if len(types) == 1:
        return types[0]
    return types


@dataclass
class IdentityFacts:
    full_name: str
    version: Optional[str] = None
    url: Optional[str] = None


@dataclass
class GrantFacts:
    public: str = "not_mentioned"
    work_signals: str | list[str] = "not_mentioned"
    grants: list[str] = field(default_factory=list)
    reserved: list[str] = field(default_factory=list)
    disclose: str = "not_mentioned"
    same_license_this: str = "not_mentioned"
    compat_names: list[str] = field(default_factory=list)
    notice: list[str] = field(default_factory=list)


@dataclass
class CombineSituation:
    id: str
    forms: list[str] = field(default_factory=list)
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    same_license: str = "not_mentioned"
    restrictions: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class CombineFacts:
    mentioned: bool = False
    situations: list[CombineSituation] = field(default_factory=list)
    other_work_must_same_license: str = "n/a"


@dataclass
class ModifyFacts:
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    same_license: str = "not_mentioned"
    restrictions: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=lambda: ["raw", "binary"])
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class AmalgamateFacts:
    """把本作品与其他作品合成一部并随新作品发布。没讨论时不写行。"""
    mentioned: bool = False
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    same_as_modify: str = "not_mentioned"
    restrictions: list[str] = field(default_factory=list)
    restrictions_stated: bool = False
    # 换证核对给出了结论。没结论时，与修改同类才继承修改行。
    relicense_settled: bool = False
    # 问卷里出现了「修改版与衍生整体」那句完整引文。
    saw_follow_sentence: bool = False
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class OutputFacts:
    mentioned: bool = False
    bound: str = "not_mentioned"
    use_rules: str = "not_mentioned"
    improve_other: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class RestrictFacts:
    """使用限制 / AUP。C 层义务，不经许可短名推断。"""
    mentioned: bool = False
    pass_on: str = "not_mentioned"
    runtime: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class NetworkFacts:
    """远程网络交互是否触发与分发相同的义务。"""
    mentioned: bool = False
    remote_same_as_convey: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class TrainFacts:
    """用本作品训练/微调（含权重迁移）。"""
    mentioned: bool = False
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class EmbedFacts:
    """嵌入/翻译/改编进另一作品，不是单纯聚合。"""
    mentioned: bool = False
    bound: str = "not_mentioned"
    result_kind: str = "not_mentioned"
    relicense: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class AugmentFacts:
    """remix / 改写 / 增广；数据许可上的 Adapted Material 非翻译部分。"""
    mentioned: bool = False
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class StatFacts:
    """统计分析 / 计算用途；仅原文讨论才建模。"""
    mentioned: bool = False
    bound: str = "not_mentioned"
    relicense: str = "not_mentioned"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class TextCues:
    """词表从全文抽出的套话/否定句，不经 LLM。"""
    rights_prefix: list[str] = field(default_factory=list)
    non_commercial: bool = False
    patent_grant: bool = False
    patent_denied: bool = False
    trademark_use_denied: bool = False
    # 明示保留商标权本身。使用权禁止不写入此项。
    trademark_right_reserved: bool = False
    waive_copyright: bool = False
    sublicense_granted: bool = False
    sublicense_forbidden: bool = False
    mark_changes: bool = False
    attach_license: bool = False
    retain_notices: bool = False
    # 源码须附完整许可、目标码分句未要求附文本 → copy 行按形态拆
    binary_skips_attach_license: bool = False
    # 同句授予「另附不同许可」且未限制新许可本身、也没有除外条款。
    different_terms_relicense: bool = False
    # 同句把该授权写给修改版，也写给 derivative/model derivatives as a whole。
    amalgamate_follows_modify: bool = False
    # 版权授予句中的权能（reproduce/prepare/distribute…），供 rights 回填。
    grant_use: bool = False
    grant_modify: bool = False
    grant_redistribute: bool = False
    grant_sublicense: bool = False
    grant_merge: bool = False
    # 许可对象定义为内容/数据库（CC Licensed Material 等），非模型权重主体。
    content_data_work: bool = False
    # 许可对象以模型权重/Checkpoint 等为主。
    model_work: bool = False
    # 许可对象的定义包含源代码、软件或用于运行模型的脚本。
    software_work: bool = False
    # 允许修改，但明文收回对衍生材料 / 抽出模型的分发。
    nd_no_redistribute: bool = False
    # 商业使用已授予，但受规模门槛或授予地域排除的限制。使用政策不算。
    commercial_use_limited: bool = False
    # 下游权利只能由原许可人另行授予，随附同一份许可不能取得权利。
    sole_source_downstream: bool = False
    # 「做你想做的」式极宽松（无再分发通知/同许可义务）。
    ultra_permissive: bool = False
    # 同许可约束范围。none=未锁定，沿用问卷；其余档位锁定装配。
    # whole=整个作品；file_all_forms=含本作品的文件在两种形态下都受治理；
    # source_only=仅一种形态必须用本许可，另一种形态可以换条款。
    same_license_scope: str = "none"
    # 「只能用本许可」的做法是附上许可文本，不是衍生作品传染。
    source_same_by_attaching: bool = False
    # 另一形态可以改用其他条款（明确写出 different terms / 另一许可）。
    binary_relicense_free: bool = False
    # 另一形态只能用「符合本许可」的条款，没有明确的任意换证。
    binary_relicense_conditional: bool = False
    # 接收者自动从原许可人获得许可。
    auto_downstream_grant: bool = False
    # sublicense 只出现在某一形态的换证条件里，不在授权句里。
    sublicense_distribution_only: bool = False
    # 使用限制句把衍生作品、输出或软件产出的数据列进限制对象。
    distill_named_restriction: bool = False
    # 授予限于某种用途，且衍生或另附条款仍须把使用维持在该用途内。
    purpose_use_passes_on: bool = False
    # 分发衍生必须标明修改，且没有把集合或聚合排除出衍生。
    derivative_must_state_changes: bool = False
    # 许可对象本身是被翻译的语言资源，且翻译被算进修改，没有单独的翻译结果。
    embed_inherits_modify: bool = False
    # 整体必须用本许可，同时例外点名的是另一类作品。
    other_kind_exception: bool = False
    # 衍生定义覆盖修改版和基于本模型的作品，且你做的衍生可以另附条款。
    your_derivative_may_relicense: bool = False


@dataclass
class CompatFacts:
    allow: list[str] = field(default_factory=list)
    ban: list[str] = field(default_factory=list)
    # 原文只允许组合、且对方作品仍守该许可。模型不得把它写进 compat。
    rejected: list[str] = field(default_factory=list)
    # 新作品整体可以采用的名字：规则已判定的，以及未决候选里模型答 yes 的。
    adopted: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class LegalIR:
    identity: IdentityFacts
    grant: GrantFacts = field(default_factory=GrantFacts)
    combine: CombineFacts = field(default_factory=CombineFacts)
    modify: Optional[ModifyFacts] = None
    amalgamate: Optional[AmalgamateFacts] = None
    output: Optional[OutputFacts] = None
    restrict: Optional[RestrictFacts] = None
    network: Optional[NetworkFacts] = None
    train: Optional[TrainFacts] = None
    embed: Optional[EmbedFacts] = None
    augment: Optional[AugmentFacts] = None
    stat: Optional[StatFacts] = None
    cues: TextCues = field(default_factory=TextCues)
    named: CompatFacts = field(default_factory=CompatFacts)
    evidence: list = field(default_factory=list)
    l0_ok: bool = False
    combine_ok: bool = False
    modify_ok: bool = False
    amalgamate_ok: bool = False
    output_ok: bool = False
    compat_ok: bool = False
    restrict_ok: bool = False
    network_ok: bool = False
    train_ok: bool = False
    embed_ok: bool = False
    augment_ok: bool = False
    stat_ok: bool = False
