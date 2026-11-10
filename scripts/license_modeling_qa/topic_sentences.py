"""主题句草稿：供向量召回 / Phase 0 观测。与问卷意图对齐，不含 SPDX 族名。"""
from __future__ import annotations

# topic -> 中英主题句（可后续按观测结果改写）
TOPIC_SENTENCES: dict[str, list[str]] = {
    "combine_agg": [
        "仅将独立作品放在同一存储或发行介质上，并不使另一作品受本许可约束。",
        "Mere aggregation of independent works on a storage or distribution medium does not bring the other work under this license.",
    ],
    "combine_other_uses": [
        "使用本作品的应用程序或其他作品可以在自选条款下分发，只要本作品部分仍遵守本许可。",
        "A work that uses this library or program may be distributed under terms of your choice, while the covered part remains under this license.",
    ],
    "combine_absorbed": [
        "组合作品作为整体必须按本许可再许可，或本许可约束整个衍生作品。",
        "The combined work as a whole must be licensed under this license, or this license covers the entire derivative work.",
    ],
    "combine_database_assimilate": [
        "将数据库实质内容纳入你拥有权利的新数据库时，新数据库整体成为改编材料。",
        "Including a substantial portion of database contents in a database where you have rights makes that database Adapted Material.",
    ],
    "copyleft_sharealike": [
        "修改版或基于本作品的作品必须以相同条款再许可。",
        "Modified versions or works based on this work must be licensed under the same terms or ShareAlike conditions.",
    ],
    "modify_adapt": [
        "允许修改、改编或制作基于本作品的衍生作品。",
        "Permission to modify, adapt, or create derivative works based on this work.",
    ],
    "use_restriction_aup": [
        "列出禁止用途或可接受使用政策，再分发时须把这些使用限制传给下游。",
        "Prohibited uses or an acceptable use policy that must be passed on when redistributing the work.",
    ],
    "compat_convert": [
        "允许新作品整体改用另一份被点名的许可，或该许可的特殊要求适用于组合整体。",
        "Permission for the combined work as a whole to adopt another named license, or for that license's special requirements to apply to the combination as such.",
    ],
    "notice_attribution": [
        "再分发时须保留版权声明、许可文本或署名通知。",
        "Redistribution must retain copyright notices, include the license text, or provide attribution.",
    ],
    "patent": [
        "本许可向你授予专利许可，或明确不授予专利权。",
        "This license grants you a patent license, or expressly states that patent rights are not licensed.",
    ],
    "network_saas": [
        "通过计算机网络远程与用户交互的修改版，必须向用户提供对应源代码。",
        "Modified versions that interact with users remotely through a computer network must offer corresponding source to those users.",
        "Making the work available over a network to others is treated as distribution under this license.",
    ],
    "output_generate": [
        "运行本作品产生的输出是否仍受本许可约束。",
        "Whether output from running the covered work is itself covered by this license.",
    ],
    "train_finetune": [
        "允许或限制用本作品训练、微调模型或迁移权重。",
        "Permission or limits on using this work to train, fine-tune, or transfer model weights.",
    ],
    "source_disclose": [
        "分发目标码时必须提供对应的完整源代码。",
        "When conveying object code, corresponding complete source code must be provided.",
    ],
    "commercial": [
        "禁止或限制商业使用，或明确允许商业用途。",
        "Commercial use is prohibited, restricted, or expressly permitted.",
    ],
}

# Phase 0 优先观测的主题（与补召回价值高的问卷包对齐）
PHASE0_TOPICS: tuple[str, ...] = tuple(TOPIC_SENTENCES.keys())
