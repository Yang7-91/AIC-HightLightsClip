# Stage 3 Diagnostic / SBC-1 — 多提示词与多视角语义边界判别探针报告

日期：2026-09-22
结论：**SBC-1 = NOT_ACTIONABLE**（三种 prompt/输入表达均不可分；不进入 BR-3）

## 1. 为什么不直接进入 BR-3

本仓库（AIC-HightLightsClip）已有过完整的 temporal refinement 失败序列：BR-1（Qwen 直接输出边界 → Recall tail 崩溃）、DTL-1（细块判定 → Precision 净损失）、SABR-1.1（显著性收缩 → Recall 损失大于收益）、BR-2（低层视觉信号 → 几乎不动作）、BHD-0.1（修复 oracle 后有理论 headroom：±1s ΔF1 +0.126，但为 non-deployable）、SBC-0（局部 clip + TRIM/KEEP/EXPAND 直接动作判断 → accuracy 37.8%、AUC 0.51-0.52、不可分）。

在 SBC-0 证明"抽象动作判断不可分"之后，直接进入 BR-3（正式部署该信号）意味着把一个已验证不可分的信号变成 refined candidates —— 不合理。SBC-1 因此设计为**任务表达方式的可分性验证**：如果换一种更适合 LLM 的表达（更清晰的动作解释 / 候选片段比较 / 核心-边界对比）仍不可分，就应该停止 prompt 小修，而不是继续投入。

## 2. SBC-0 可能失败的原因与 SBC-1 假设

SBC-0 的失败可能来自任务表达而非模型能力：
- 直接问局部 clip "TRIM/KEEP/EXPAND"过于抽象；
- 模型看不到候选核心事件，不知道边界在裁什么；
- LLM 通常更擅长**比较**而不是输出抽象标签。

SBC-1 用三种预注册表达验证该假设（运行前冻结，运行后未改 prompt）：

| 变体 | 表达方式 | 输入 |
|---|---|---|
| P1_ACTION_V2 | 改良动作分类：明确左右方向的 TRIM/KEEP/EXPAND 定义、事件完整性、上下文冗余、保留动作起因/主体/结果 | 边界 ±4s clip |
| P2_VARIANT_RANKING_V1 | 候选片段比较：给 KEEP / TRIM / EXPAND 三个版本的片段（边界位移 1.5s），问哪个最完整覆盖事件且含最少无关上下文 | 同一视频的 3 个 4s clip |
| P3_CORE_CONTRAST_V1 | 核心-边界对比：先给候选核心 clip（中间 4s），再给边界 clip，问边界外侧是否仍属同一事件 | 2 个 clip |

## 3. 数据与样本

- oracle labels：BHD-0.1 产出 `boundary_oracle_labels.jsonl`（weak-reference 派生的 F1-optimal optional oracle 标签；仅用于诊断评估，不进入任何 prompt payload）
- candidate/视频元数据：Stage 4.2 frozen candidate cache（只读）+ Stage 1 dev 视频清单
- 视频：Dev 166 个 clip 全部就绪（沿用已校验下载）
- 样本：**240 基础边界侧样本**（left 120 / right 120；TRIM 80 / KEEP 80 / EXPAND 80；每侧每类 40 选满；seed 20260922；零跳过）× 3 变体 = **720 条诊断样本**
- 每个样本只注入确定性上下文（side/候选时长/边界位置/方向说明）；**不把 oracle label 发给模型**

## 4. 模型设置

- model：**Qwen/Qwen3.5-4B**（AutoDL vLLM，OpenAI 兼容 API）
- base_url：`http://127.0.0.1:8000/v1`；temperature 0.0；top_p 1.0；max_tokens 512；response_format json_object
- **enable_thinking=false**（chat_template_kwargs；与 SBC-0 相同的 JSON 任务必要条件）
- 视频输入：ffmpeg 截取片段 → `data:video/mp4;base64` data URL（fps 2.0）
- qwen_calls / vllm_calls = **720 / 720**（P1/P2/P3 各 240）

## 5. 结果（720 样本，weak-reference oracle 标签为诊断目标）

| 变体 | schema rate | accuracy | macro-F1 | left AUC-like | right AUC-like | decision |
|---|---|---|---|---|---|---|
| P1_ACTION_V2 | **1.000** | 0.325 | **0.185** | 0.513 | 0.510 | NOT_ACTIONABLE |
| P2_VARIANT_RANKING_V1 | **0.729** | 0.240 | **0.203** | 0.515 | 0.523 | NOT_ACTIONABLE |
| P3_CORE_CONTRAST_V1 | **0.992** | 0.328 | **0.172** | 0.501 | 0.516 | NOT_ACTIONABLE |

Per-class（precision / recall）：

| 变体 | TRIM | KEEP | EXPAND |
|---|---|---|---|
| P1 | 0.328 / **0.938** | 0.143 / 0.013 | 0.500 / 0.025 |
| P2 | 0.212 / 0.757 | 0.326 / 0.241 | 0.000 / 0.000 |
| P3 | 0.329 / **0.963** | 0.000 / 0.000 | 0.250 / 0.013 |

Confusion（行=oracle，列=预测）：
- P1：TRIM [75,4,1] / KEEP [78,1,1] / EXPAND [76,2,2]
- P2：TRIM [28,9,0] / KEEP [44,14,0] / EXPAND [60,20,0]
- P3：TRIM [77,0,3] / KEEP [80,0,0] / EXPAND [77,0,1]

**总 summary**：best_variant = P2_VARIANT_RANKING_V1（macro-F1 最高 0.203，仍远低于 0.45）；decision = **NOT_ACTIONABLE**；recommendation = **FREEZE_TEMPORAL_OR_TRY_COARSE_RETRIEVAL_RESCORING**。

## 6. 失败机制

1. **三种表达产生同一种退化行为**：模型强烈偏向 TRIM（P1/P3 中 93-96% 的预测是 TRIM；P2 中 EXPAND 从未被选择）—— 改良版 prompt 强调"边界含多余上下文则 TRIM"后，模型从 SBC-0 的"保守 KEEP"倒向"一律 TRIM"，但无论如何都与 oracle 标签无关；
2. **AUC-like 0.50-0.52**：预测置信度对 oracle 动作没有排序能力，接近随机；
3. **P2 比较形式没有帮助**：即使把任务改成"三版本片段比较"（LLM 更擅长的形式），EXPAND recall 仍为 0，schema rate 反而降到 0.729（比较任务的输出格式更复杂）；
4. **结论**：当前 4B VLM + 单帧率局部片段输入对 weak-reference 派生的边界动作标签**没有可分性**；这不是 prompt 表达问题，继续 prompt 小修没有依据。

## 7. 与 SBC-0 的关系

SBC-0（单 prompt，180 样本）：accuracy 37.8%、macro-F1 0.350、AUC 0.51-0.52，强 KEEP 偏好。
SBC-1（三 prompt，720 样本）：accuracy 24-33%、macro-F1 0.17-0.20、AUC 0.50-0.52，强 TRIM 偏好。
—— 两种任务表达、两种偏置方向，相同结论：**不可分**。SBC 系列探针至此完成负结果闭环。

## 8. 是否 actionable / 是否进入 BR-3

- 是否有 ACTIONABLE_SIGNAL：**无**（三变体均未达到 schema 0.95 + macro-F1 0.45 + AUC 0.65）
- 是否进入 BR-3：**不进入**（本轮为 diagnostic-only，未生成任何 refined candidates / 正式预测）

## 9. 隔离性

- Hard：未运行；Heldout：未访问；Audit36：未使用
- 未训练 / 未微调模型；未下载新模型
- 未修改 Stage 2 / Stage 3 正式主线（本仓库 v2 的 prompt/parser/merge 主逻辑零改动；SBC-1 为独立 diagnostic 分支）
- 未修改 frozen cache 或既有实验产物；oracle labels 未写入正式预测
- 视频 / 模型 / runs / outputs 未上传 GitHub（仅代码、配置、prompt、测试与报告入库）

## 10. 实现与运行记录

- 新增：`v2/configs/stage3/sbc1_semantic_boundary_probe.yaml`、`v2/prompts/stage3_sbc1_{p1,p2,p3}_*.md`、`v2/src/video_highlight/stage3_boundary_refine/sbc1_probe.py`、`v2/scripts/run_sbc1_probe.py`、`v2/tests/stage3/test_sbc1_probe.py`；
- 测试：定向 `python -m unittest discover -s v2/tests/stage3 -v` → **20 passed**；全仓 `unittest discover -s v2/tests` → **58 passed**；
- 运行环境：AutoDL（Qwen/Qwen3.5-4B vLLM 0.28.0），输出目录 `/root/autodl-tmp/outputs/stage3_sbc1_probe_20260922_121051/`（未上传）；
- 过程修复（已提交）：CLI `--config` 位置兼容、ffmpeg mp4 管道输出改临时文件、采样排序 bug；
- 运行全程无 prompt 修改；一次数据链路 smoke（3 样本）通过后跑全量。

## 11. 下一步建议（本轮不执行）

SBC-1 全负后，建议按优先级选择：

1. **CRR-1 — Coarse Retrieval Re-ranking**（推荐）：不再改边界，在 Stage 2 层面让 Qwen 对候选做整段高光评分 / 完整高光判断（不输出时间戳），与候选 score fusion，目标是减少低价值候选而非裁剪边界；
2. **MCG-1 — Multi-Candidate Generation Self-Consistency**：多固定 prompt 生成候选，取高一致性区间，低一致性区间保留高召回 fallback；
3. **冻结 temporal refinement**：接受当前边界质量，把精力转向 Stage 5 spatial / 提交策略；
4. 不建议：继续在 SBC prompt 表达上小修（已有 2 轮、4 种表达、900 样本的负结果）；不建议直接设计 BR-3。
