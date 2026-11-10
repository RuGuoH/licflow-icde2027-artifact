"""问卷题干。选项是法律事实，不含 ModelGo 字段名与许可族名。"""
from __future__ import annotations

from .types import EvidenceItem

SYSTEM = (
    "你是许可证文本阅读器。只根据给定引文回答。只输出一段 JSON 对象，不要 Markdown，"
    "不要解释。没有引文支持时不得选 yes。不得根据许可证名称推断。"
)

_L0_SPEC = """
请根据引文回答下列问题。JSON 的每个键对应一题，值为 {{"value": ..., "evidence_ids": ["e1"]}}。
多选题的 value 为字符串数组；单选题的 value 为字符串。

L0_public: 该许可是否向不特定公众提供（而非仅给特定合同相对方）？
  选项: yes | no | not_mentioned

L0_work_signals: 原文把哪几类作品定为许可对象（可多选）？
  选项: software, data, model
  value 为字符串数组。都没有定义句时 value 为 []。
  被授权对象包含程序源码、软件或用于运行模型的脚本 → software。
  被授权对象包含数据库、语料、数据集，或 CC 式 Licensed Material（artistic/literary work, database）→ data。
  被授权对象包含模型权重、Checkpoint、以 Model 为授权对象 → model。
  定义句同时包含模型和源代码时，software 与 model 都要选。
  只写 research materials、没有上述定义句的，不要选。
  不要因为正文出现 software 这个词，就把模型权重许可选成只有 software。

L0_grants: 原文明确授予哪些权利（可多选）？
  选项: use, modify, merge, redistribute, sublicense, commercial_use, patent_use
  映射（版权授予句，不是使用限制清单）：
    reproduce / use the Model or work → use；
    prepare Derivatives / modify / adaptations → modify；
    distribute / Distribution / publish → redistribute；
    sublicense → sublicense；
    merge（MIT 等字面）、明确允许与他作组合，或版权授予句 prepare/distribute Derivatives
      （合入他作权能，即使下游有 Use-based restrictions）→ merge。
  有「hereby grants … copyright license to reproduce, prepare, … distribute」须勾上对应项
  （含 merge，若授予制备/分发 Derivatives）。
  不得因后文 Use-based restrictions / Attachment 禁止某些用途，就拒勾 use。
  若未授予任何一项，value 为 []

L0_reserved: 原文明确保留或禁止哪些（可多选）？
  选项: trademark_use, copyright, patent, trademark, sublicense, commercial_use, redistribute
  若无，value 为 []

L0_disclose: 分发二进制/目标码时，是否必须提供对应源代码？
  选项: yes | no | not_mentioned
  「Corresponding Source」「Executable Form … Source Code Form」
  「object code / binary 分发须提供 source」「made available as Source Code」→ yes。
  仅定义「Source Code means…」、或仅「network server at no charge」作为交付渠道 → 不因此单独答 yes。

L0_same_license_this: 对本作品的修改或衍生再分发时，是否必须使用本许可（ShareAlike / copyleft），或原文列出的兼容许可？
  选项: yes | no | not_mentioned
  「modified versions must be licensed under this License」「share alike」「license the entire work under this License」→ yes。
  「修改版 / work based on … under the terms of this License 或 under the terms of Section N」→ yes。
  「source code form … only under this license」→ yes。
  「additional or different license terms」「released under different licensing terms」→ no（允许换许可，不是 copyleft）。
  仅「give / provide / include / distribute a copy of this License」、署名、this list of conditions、
  endorse/promote/广告命名限制、或使用限制传给下游 → no（那是通知/AUP，不是 copyleft）。
  「Adapter's License」且没有 ShareAlike、只要求不得妨碍下游遵守本许可 → no。
  mere aggregation / Collection 收录本身 → 不因此答 yes。

L0_compat_names: 不要猜测 SPDX 短名。本题 value 一律 []。
  原文点名可转换到的其他许可，不在 L0 填写。

L0_notice: 再分发须附带什么（可多选）？
  选项: notice, license_text, state_changes
  notice = 保留版权声明、指向本许可的声明、免责声明等（若许可方提供）。
  license_text = 标明以本许可授权，并附许可全文，或 URI/超链接到本许可。
  若原文同时要求「保留声明」与「标明本许可（含 URI）」，必须同选 notice 与 license_text。
  state_changes = 须表明是否修改过作品，或保留曾修改的标记；
    含「notice/inform … that you modified」类表述时勾选。
  若无或未提及，value 为 []
"""

_COMBINE_SPEC = """
以下问题只关于「未修改本作品、但把它和其他作品放在一起」。不要回答对本作品本身的修改。
引文已按情况分组。C2/C3 在对应组有引文时，禁止用 not_mentioned 回避。

C0: 原文是否讨论这种「放在一起」？
  选项: yes | no
  选 yes 必须提供 evidence_ids。

C1: 若 C0 为 yes，原文区分了哪几种情况（可多选）？有对应分组引文就必须勾。
  选项:
    agg — 仅因存放/发行在一起，仍分离、不构成一部程序
    other_uses — 另一作品使用/链接/调用本作品，但不是对本作品的修改
    absorbed — 放在一起后法律上视为一部作品、整体受本许可约束
    db_assimilate — 把本作品或数据库内容的实质部分纳入另一部作品/数据库，使后者成为改编/衍生对象
    mixed_type — 明确谈与不同类型作品放在一起（语言资源/模型/数据与其他类型聚合）
  有【不同类型组合】引文就必须勾 mixed_type。
  有【数据库/内容融入】引文就必须勾 db_assimilate。
  「mere aggregation of another work not based on the linguistic resource」→ mixed_type，不是 other_uses。
  仅改编本作品本身（Adapted Material 定义、remix）→ 不要勾 db_assimilate，那是 Modify 卡。
  C0 为 no 时 value 为 []。

C2: 仅当 C1 含 other_uses：另一作品本身是否必须以本许可（或兼容许可）再授权？
  选项: yes | no | n/a
  只看【另一作品使用/链接】组；禁止用【聚合】组的 mere aggregation / does not cause
  this License to apply / does not bring 来回答 C2（那是 C4_agg_bound）。
  规则（有引文时不得选 not_mentioned，也不得用 n/a 代替）：
    - 【使用/链接】出现 terms of your choice / work that uses … under terms of your choice
      → no（即使【吸收】组同时有 entire work）
    - 【使用/链接】没有上一句，且【使用/链接】或【吸收】出现
      license the entire work … under this License → yes
    - agg 是否独立只看 C4_agg_bound，不要和 C2 混成一道题。
  若不问此题，value 为 n/a。

下列键一律扁平，不要嵌套对象。C1 未勾选的情况填 n/a。

C3_agg / C3_other_uses / C3_absorbed / C3_db_assimilate / C3_mixed_type:
  该情况适用的提供形态。选项: raw | binary | saas | same_all | n/a
  有引文时不要填 not_mentioned。若原文未按形态区分：
    agg 填 binary；other_uses 与 absorbed 填 raw；db_assimilate 填 same_all。

C4_agg_bound / C4_other_uses_bound / C4_absorbed_bound / C4_db_assimilate_bound / C4_mixed_type_bound:
  该情况下产物是否仍须遵守本许可。选项: bound | not_bound | n/a
  agg 有「mere aggregation / does not bring」→ not_bound。
  mixed_type 有「mere aggregation / does not bring / not based on the linguistic resource」→ not_bound。
  absorbed 有「entire work as a whole」→ bound。
  db_assimilate 有「database … Adapted Material / substantial portion … contents」→ bound。

C4_agg_relicense / C4_other_uses_relicense / C4_absorbed_relicense / C4_db_assimilate_relicense / C4_mixed_type_relicense:
  产物再分发时能否改用不同许可。选项: yes | no | conditional | n/a
  db_assimilate：Adapter's License「不得阻碍下游遵守本许可」→ conditional；ShareAlike / Compatible License → conditional。

C4_agg_restrictions / C4_other_uses_restrictions / C4_absorbed_restrictions / C4_db_assimilate_restrictions / C4_mixed_type_restrictions:
  分发义务，可多选 include_notice, include_license, state_changes；无则 []；不问则 n/a。
"""

_MODIFY_SPEC = """
只问「修改本作品本身」，不要问与其他作品的组合。

T1_bound: 修改后的作品是否仍须遵守本许可？
  选项: bound | not_bound | not_mentioned

T1_relicense: 修改后的作品再分发时，能否改用与本许可不同的许可？
  选项: yes | no | conditional | not_mentioned
  「Adapter's License」且没有 ShareAlike：只要求不得妨碍下游遵守本许可 → no（不是 conditional）。
  「may provide additional or different license terms」用于修改版/衍生 → yes。
  「license of your choice that does not contradict」「compatible license」「Secondary License」→ conditional。
  ShareAlike / Compatible License → conditional 或随同许可规则。

T1_same_license: 修改后的作品是否必须以本许可（或原文列出的兼容许可）再授权？
  选项: yes | no | not_mentioned
  「share alike」「license the entire work under this License」→ yes。
  「修改版须 under the terms of this License / Section N」→ yes。
  「additional or different license terms」「give / provide a copy of this License」→ no。
  「Adapter's License」且没有 ShareAlike → no。

T1_restrictions: 分发修改作品时的附带义务（可多选）
  选项: include_notice, include_license, state_changes
  保留版权/专利/商标或归因通知 → include_notice。
  须随附本许可文本 → include_license。
  须标明已修改（notice/inform recipients that you modified、prominent notices stating changes）
    → state_changes；可与 include_notice 同时勾选。
  若无，value 为 []

T1_forms: 上述义务适用的提供形态（可多选）
  选项: raw, binary, saas
  未区分则 value 为 ["raw","binary"]
"""

_OUTPUT_SPEC = """
只问「运行本作品产生的输出/结果」，以及能否用这些输出去改进另一模型。不要问对本作品本身的微调。

O0: 原文是否讨论运行本作品所产生的输出或结果？
  选项: yes | no
  选 yes 必须提供 evidence_ids。

O1: 若 O0 为 yes，该输出本身是否作为 covered work / Model Derivative 等整包必须遵守本许可？
  选项: bound | not_bound | only_if_output_is_covered | n/a
  「only if the output … constitutes a covered work」→ only_if_output_is_covered。
  「outputs by themselves are not deemed Model Derivatives」或「Licensor claims no rights in the Output」
    → not_bound（这只回答作品归属/是否整包受许可，不等于使用不受限制）。
  有 covered-work 句时不得选 bound。
  O0 为 no 时本键填 n/a。

O2: 输出的使用是否还必须遵守原文另行规定的使用限制或运行时规则？
  选项: yes | no | n/a
  这里不问输出本身是否构成受保护作品、是否整包受本许可。那是 O1。
  下列任一条 → yes：
    「No use of the output can contravene any provision」；
    输出的后续使用须遵守 use-based restrictions / acceptable use / Attachment 中的使用限制；
    同句同时出现 output 与 use-based restrictions。
  下列情况 → no：
    原文只规定「输出仅在其本身包含本作品的可版权内容、或本身构成基于本作品的作品时，才受本许可约束」；
    只有「不主张 Output 权利 / claims no rights」，没有上述使用限制句。
  不得因 O1=not_bound 或 O1=only_if_output_is_covered 就改判；O2 只看有没有单独的使用限制句。
  O0 为 no 时本键填 n/a。O0 为 yes 且没有上述使用限制句时填 no，不要填 n/a。

O3: 是否允许把本作品的输出、中间表示或权重用来训练、改进或蒸馏另一模型？
  选项: yes | no | not_mentioned | n/a
  「will not use … output … to improve any other large language model」→ no。
  「distillation methods」只是把蒸馏定义为衍生，不是禁止，不要因此选 no。
  没有这类句子 → not_mentioned。
  既无输出线索也无蒸馏线索时填 n/a。
"""

_TRAIN_SPEC = """
只问「训练/微调本作品，或把本作品的权重迁移到另一模型」。不要问运行输出，不要问嵌入文本数据。

M0: 原文是否讨论训练、微调本作品，或把本作品的权重/参数迁移到另一模型？
  选项: yes | no
  fine-tune / finetuning / transfer of patterns of the weights / Use may include … training → yes。
  仅把 “trained model weights”“training-enabling code” 当作作品清单、没有训练行为 → no。
  选 yes 必须提供 evidence_ids。

M1: 若 M0 为 yes，训练/微调（或权重迁移）得到的产物是否必须遵守本许可？
  选项: bound | not_bound | n/a
  “Derivatives of the Model / Model Derivatives” 包含 transfer of weights、finetuning、distillation methods → bound。
  明确训练产物不受本许可约束 → not_bound。
  M0 为 no 时填 n/a。

M2: 若 M0 为 yes，该产物再分发时能否改用不同许可？
  选项: yes | no | conditional | n/a
  M0 为 no 时填 n/a。
"""

_EMBED_SPEC = """
只问数据侧再利用：嵌入/翻译、remix/增广、统计分析。不要问 mere aggregation（那是 Combine），不要问训练模型。

E0: 原文是否讨论把本作品嵌入另一作品、翻译、或产生 Adapted Material（含 translated）？
  选项: yes | no
  Adapted Material / licensed material is translated / linguistic resource 被改写进另一作品 → yes。
  mere aggregation / 单纯存放在一起 → no。
  选 yes 必须提供 evidence_ids。

E1: 若 E0 为 yes，嵌入/改编后的作品是否仍须遵守本许可？
  选项: bound | not_bound | translation | n/a
  改编须 share-alike / 相同许可 → bound。
  主要当作翻译且未要求整份作品改用本许可 → translation。
  E0 为 no 时填 n/a。

E2: 若 E0 为 yes，嵌入/改编结果再分发时能否改用不同许可？
  选项: yes | no | conditional | n/a
  E0 为 no 时填 n/a。

A0: 原文是否讨论 remix / 改写 / 增广，或 Adapted Material 中的 altered / arranged / transformed（不是单纯翻译）？
  选项: yes | no
  仅翻译、mere aggregation、训练模型 → no。
  选 yes 必须提供 evidence_ids。

A1: 若 A0 为 yes，增广/remix 后的作品是否仍须遵守本许可？
  选项: bound | not_bound | n/a
  Adapted Material / remix 仍受本许可约束 → bound。
  A0 为 no 时填 n/a。

A2: 若 A0 为 yes，该产物再分发时能否改用不同许可？
  选项: yes | no | conditional | n/a
  A0 为 no 时填 n/a。

S0: 原文是否讨论把本作品用于统计分析、计算用途，或明确说结果本身不是语言资源？
  选项: yes | no
  仅有 ordinary use / 运行输出 / 训练 → no。
  选 yes 必须提供 evidence_ids。

S1: 若 S0 为 yes，统计/计算结果是否仍须遵守本许可？
  选项: bound | not_bound | n/a
  「not itself a linguistic resource」或统计结果不受本许可约束 → not_bound。
  S0 为 no 时填 n/a。

S2: 若 S0 为 yes，该产物再分发时能否改用不同许可？
  选项: yes | no | conditional | n/a
  S0 为 no 时填 n/a。
"""

_COMPAT_SPEC = """
只问原文是否点名另一份许可。不要根据本许可的名称推断兼容关系，不要填写族谱。
判断对象是组合或融合之后的新作品整体能否采用该许可。原来各部分仍守原许可，不能据此答 yes。

K0: 原文是否允许新作品整体采用另一份被点名的许可？
  选项: yes | no
  修改版或新作品可以改用该许可，或该许可的要求适用于组合整体 → yes。
  仅有 permission to link or combine，且对方作品 remain governed by 该许可 → no。
  前言里「若你想做库，去用另一份许可」不是授权，选 no。

K1: 若 K0 为 yes，从引文抄出那份许可的原文称呼和版本（字符串数组）。不要改写成短 ID。
  K0 为 no 时 value 为 []。

K2: 原文是否点名禁止与某份许可共用或再许可？从引文抄出称呼。没有则 []。
  「incompatible with the aim of …」不是点名另一许可。
"""

_ADOPT_SPEC = """
下面每条候选都来自本许可原文。只判断：把本许可的作品与该候选许可的作品组合或融合后，新作品整体是否可以采用该候选许可。
原来各部分仍守各自的许可。不要用这一点代替对「新作品整体」的判断。不要根据许可名称推断。

- 引文写明新作品或修改版可以改用该候选许可，或该候选许可的要求适用于组合整体 → adopt = yes
- 同一段里既写本许可继续适用于原来那一部分，又写候选许可的要求适用于组合整体 → adopt = yes
- 引文只允许组合，并写明对方作品仍受该候选许可约束、没有把候选许可的要求放到组合整体上 → adopt = no
- 引文没有写明新作品整体可以采用该候选许可 → adopt = no

只对列出的候选作答。candidate 必须原样抄写给出的短名。不要新增候选。
JSON: {"items": [{"candidate": "短名", "adopt": "yes"}]}
adopt 只能是 yes 或 no。
"""

_RESTRICT_SPEC = """
只问「禁止用途 / 可接受使用 / 使用限制清单」。不要问版权 copyleft，不要问组合。

R0: 原文是否列出禁止用途、可接受使用政策、或使用限制清单？
  选项: yes | no
  「use-based restrictions」「acceptable use」「prohibited uses」
  「You agree not to use … for any illegal / harm」→ yes。
  「ResponsibleAI / Responsible AI」且后文列禁止用途 → yes。
  仅有 warranty / as is、或「不得再施加 further restrictions」不是本题，选 no。
  选 yes 必须提供 evidence_ids。

R1: 若 R0 为 yes，再分发本作品或其衍生时，是否必须把这些使用限制一并施加给下游？
  选项: yes | no | n/a
  「MUST be included as an enforceable provision」「shall require all of your users」
  「include the use restrictions」→ yes。
  「不得附加与本许可不一致的条款」「offer or impose any terms … inconsistent with」
  「pass on these use restrictions」→ yes（使用限制传递，不是 ShareAlike/copyleft）。
  R0 为 no 时填 n/a。

R2: 若 R0 为 yes，运行本作品（包括作为服务运行、生成内容）时是否仍须遵守这些使用限制？
  选项: yes | no | n/a
  使用限制覆盖 running / creating content with / 运行时 → yes。
  R0 为 no 时填 n/a。
"""

_NETWORK_SPEC = """
只问「通过网络远程交互本作品」或「经网络向他人提供使用且视同分发」。不要把「从网络服务器提供对应源码」当成远程交互义务。

N0: 原文是否讨论用户通过网络远程交互/使用本作品，或经网络向他人提供本作品（而非仅本地复制或源码下载）？
  选项: yes | no
  「remote network interaction」「interacting with it remotely through a computer network」→ yes。
  「hosted service」「API-based access」「remotely accessible service」→ yes（作为提供形态，不是源码披露）。
  「made available … over a network」且须按分发履行义务 / 「treat … as a distribution」→ yes。
  仅有 corresponding source / network server at no charge → no。
  选 yes 必须提供 evidence_ids。

N1: 若 N0 为 yes，远程网络交互或经网络提供是否触发与分发/传达相同的义务？
  选项: yes | no | only_if_modified | n/a
  「if you modify the Program, your modified version must … offer all users interacting」
  → only_if_modified。
  无条件对远程用户提供源码，或明文将网络提供视同分发 → yes。
  N0 为 no 时填 n/a。
"""


def _is_fulltext_item(item: EvidenceItem) -> bool:
    return item.id == "e_full" or (item.cue or "") == "fulltext"


def _fulltext_notice() -> str:
    return (
        "本许可已提供全文。未在正文出现的主题选 no / not_mentioned / []，"
        "不得推断常见开源义务。引用请使用 e_full。\n\n"
    )


def format_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "（无预抽引文。若无法从题目本身判断，一律 not_mentioned / []。）"
    if len(items) == 1 and _is_fulltext_item(items[0]):
        return f"- {items[0].id} [fulltext]\n{items[0].quote}"
    lines = []
    for e in items:
        lines.append(f"- {e.id} [{e.topic}] {e.quote}")
    return "\n".join(lines)


_COMBINE_GROUPS = (
    ("【聚合/仅存放】勾 agg。", {"combine_agg", "exception_or_separate_work"}),
    ("【另一作品使用/链接】勾 other_uses；C2 主要看这组。", {"combine_other_uses", "combine_link"}),
    ("【视为一部作品】勾 absorbed。", {"combine_absorbed"}),
    (
        "【数据库/内容融入】勾 db_assimilate。",
        {"combine_database_assimilate", "adapt_share_conditions"},
    ),
    ("【不同类型组合】勾 mixed_type。", {"combine_mix"}),
)


def format_combine_evidence(items: list[EvidenceItem]) -> str:
    if not items:
        return "（无预抽引文。若无法判断，C0 填 no。）"
    if len(items) == 1 and _is_fulltext_item(items[0]):
        return format_evidence(items)
    used: set[str] = set()
    blocks: list[str] = []
    for title, topics in _COMBINE_GROUPS:
        group = [e for e in items if e.topic in topics]
        if not group:
            continue
        used.update(e.id for e in group)
        lines = [title]
        for e in group:
            lines.append(f"- {e.id} [{e.topic}] {e.quote}")
        blocks.append("\n".join(lines))
    rest = [e for e in items if e.id not in used]
    if rest:
        lines = ["【其他】"]
        for e in rest:
            lines.append(f"- {e.id} [{e.topic}] {e.quote}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _wrap_user(
    license_name: str,
    evidence_block: str,
    spec: str,
    *,
    fulltext_mode: bool = False,
    id_note: str = "不得据此填写答案",
) -> str:
    head = f"许可标识（{id_note}）: {license_name}\n\n"
    notice = _fulltext_notice() if fulltext_mode else ""
    return f"{head}{notice}引文:\n{evidence_block}\n\n{spec.strip()}\n"


def build_l0_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _L0_SPEC,
        fulltext_mode=fulltext_mode,
        id_note="仅用于 short_id，不得据此填写其他答案",
    )


def build_combine_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_combine_evidence(evidence),
        _COMBINE_SPEC,
        fulltext_mode=fulltext_mode,
    )


_AMALGAMATE_SPEC = """
只问「把本作品和其他作品合成一部新作品，并且各部分随这部新作品一起发布」。
不要回答单纯放在一起、链接或各自独立发行（那是组合）。
不要回答只修改本作品自己、不和其他作品合成（那是修改）。

F0: 原文是否把这种合成当作本许可要管的对象？
  选项: yes | no
  衍生作品、修改版的整体、由本作品和其他材料构成的一部作品 → yes。
  同一句既写修改版，又写 derivative works、derivatives of the model 或 model derivatives as a whole → yes。
  mere aggregation、各自独立、只链接且不构成一部作品 → no。
  只定义如何修改本作品、完全没有谈到和其他作品合成 → no。
  选 yes 必须提供 evidence_ids。

F1_bound: 若 F0 为 yes，合成后的新作品是否仍须遵守本许可？
  选项: bound | not_bound | not_mentioned
  F0 为 no 时填 not_mentioned。

F2_relicense: 若 F0 为 yes，这部新作品再分发时能否改用与本许可不同的许可？
  选项: yes | no | conditional | not_mentioned
  判断顺序与修改后的作品相同：
    必须按本许可或作为整体按本许可发布 → no。
    新条款不得冲突、必须写入使用限制，或只能换成兼容许可 → conditional。
    可以另用条款，且新条款本身没有上述限制 → yes。
  原文没说 → not_mentioned。不要因为修改题的答案直接抄过来。
  F0 为 no 时填 not_mentioned。

F3_restrictions: 分发这部新作品时的附带义务（可多选）
  选项: include_notice, include_license, state_changes
  没有则 value 为 []。使用限制不要在这里回答。
  F0 为 no 时 value 为 []。

F4_same_as_modify: 原文是否用同一个定义覆盖「修改本作品」和「这种合成整体」？
  选项: yes | no | not_mentioned
  同一句或同一个定义同时写到修改版和作品整体 / derivative works as a whole → yes。
  F0 为 no 时填 not_mentioned。
"""


def build_amalgamate_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _AMALGAMATE_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_modify_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _MODIFY_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_output_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _OUTPUT_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_compat_adopt_user(license_name: str, items: list[tuple[str, str]]) -> str:
    blocks = []
    for sid, quote in items:
        blocks.append(f"候选: {sid}\n引文: {quote}")
    body = "\n\n".join(blocks) if blocks else "（无候选）"
    return (
        f"许可标识（不得据此填写答案）: {license_name}\n\n"
        f"{body}\n\n"
        f"{_ADOPT_SPEC.strip()}\n"
    )


def build_compat_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _COMPAT_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_restrict_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _RESTRICT_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_network_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _NETWORK_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_train_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _TRAIN_SPEC,
        fulltext_mode=fulltext_mode,
    )


_RELICENSE_ONLY_SPEC = """
只回答换证，并引用你用来下结论的那一句原句。不要判断产物是否仍受本许可约束。

relicense: 这一类产物再分发时，能否改用不同许可？
  选项: yes | no | conditional | not_mentioned
  按顺序命中即停：
    1. 产物本身必须按本许可、作为整体按本许可、或只能按本许可发布 → no。
       附上文本、保留声明、标明修改不是 no。
    2. 新条款本身不得冲突、必须写入使用限制，或只能换成兼容许可或 Secondary License → conditional。
    3. 可以另用条款，且新条款本身没有第 2 条那种限制 → yes。
       “your use otherwise complies with this License” 属于这句，回答 yes。
    4. 授予是 sublicensable，或 without restriction，且分发条件只有附上文本、保留声明、标明修改 → yes。
       没有字面写「可以另用条款」时，也按这一条回答 yes，不要答 not_mentioned。
    5. 不可再许可，且没有明确允许另用条款 → no。明确允许时以允许为准。
    6. 以上都没有 → not_mentioned。
  quote 必须是上面某一条引文的原文。
  只输出 JSON：{"relicense": "yes" 或 "no" 或 "conditional" 或 "not_mentioned", "quote": "原句"}
"""


def build_relicense_quote_check_user(kind: str, quote: str) -> str:
    """只用模型引用的那一句，按同一套情形再判一次。"""
    if kind == "modify":
        scope = "这一类产物是：你做出的修改版或衍生作品。"
    elif kind == "train":
        scope = "这一类产物是：你训练或微调得到的模型。"
    elif kind == "amalgamate":
        scope = "这一类产物是：把本作品与其他作品合成的一部新作品。"
    else:
        raise ValueError(kind)
    return (
        scope
        + "\n下面只有一句许可原文。用同一套规则判断这一句。\n"
        + "如果这句属于不能用来回答的那几类，回答 not_mentioned。\n"
        + "句子:\n"
        + quote.strip()
        + "\n\n"
        + _RELICENSE_ONLY_SPEC.strip()
        + "\n"
    )


def build_relicense_only_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    kind: str,
    fulltext_mode: bool = False,
) -> str:
    if kind == "modify":
        scope = "这一类产物是：你做出的修改版或衍生作品。不要判断训练或蒸馏。"
    elif kind == "train":
        scope = (
            "这一类产物是：你训练或微调得到的模型。"
            "微调若落在修改或衍生的定义里，并且原文没有单独排除微调，就按该定义的分发条件判断。"
            "数据集另有许可、只把蒸馏排除出衍生的句子，不能单独作为 no 的依据。"
        )
    elif kind == "amalgamate":
        scope = (
            "这一类产物是：把本作品与其他作品合成的一部新作品。"
            "不要用单纯放在一起或只修改本作品自己的句子代替。"
        )
    else:
        raise ValueError(kind)
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        scope + "\n" + _RELICENSE_ONLY_SPEC,
        fulltext_mode=fulltext_mode,
    )


def build_embed_user(
    license_name: str,
    evidence: list[EvidenceItem],
    *,
    fulltext_mode: bool = False,
) -> str:
    return _wrap_user(
        license_name,
        format_evidence(evidence),
        _EMBED_SPEC,
        fulltext_mode=fulltext_mode,
    )
