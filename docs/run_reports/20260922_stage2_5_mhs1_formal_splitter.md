# Stage 2.5 / MHS-1 — Multi-Highlight Candidate Splitter 正式实验报告

日期：2026-09-22
结论：**MHS-1 Dev result = NEGATIVE_RESULT / NO_PROMOTION**

## 1. 为什么 MHS-VIS 后回到正式指标实验

- MHS-VIS-0/1 可视化确认了现象：11 个典型 Dev 候选中 8 个存在 ≥2 个运动峰（"一个大段包含多个高光事件"）；
- 但可视化只是 diagnostic，不产生可提交的候选；本轮把该现象转化为 deployable 的候选拆分实验，并用 weak-reference 指标（事后评估）验证是否真的提升 Precision / F1 / tIoU。

## 2. 方法（MHS-1）

对每个 Stage 2 候选（只处理时长 ≥12s）：
1. **eventness 曲线**（1s bin，2 fps 采样）：frame difference（权重 0.45）+ shot boundary（直方图差异自适应阈值，0.20）+ audio energy（ffmpeg RMS，0.20）；无音轨时自动降权；3s 平滑 + minmax 归一化；
2. **峰-谷分析**：局部峰（间隔 ≥ peak_min_distance、高于 max(0.5·max, median)）；相邻峰间找低谷（须 ≤ valley 分位阈值）；
3. **谷区丢弃式拆分**：在显著低谷处切开，每个切点两侧共丢弃 `merge_gap_sec` 的谷区内容（去"普通内容"），子段保留事件内容；
4. **guards**：峰数 <2 → identity；谷不明显 → identity；子段数 2..max；每段 ≥ min_duration；丢弃短段后仍须 ≥2 段（never_drop_all_subsegments）；总覆盖率下降 ≤ cap；切到空/越界 → fallback identity；所有决策**不使用 weak-reference**。

预注册配置：C1（保守：间隔 4s/谷 0.40/段 4s/丢弃 2s/cap 0.35）、C2（中等：3s/0.45/3s/1.5s/0.45）、C3（更保守稳定：5s/0.35/5s/2s/0.30）。

## 3. 数据与运行

- Stage 2 candidates：frozen candidate cache（**仅 dev split**；heldout 拒绝加载）；视频：Dev 166 个 clip 全就绪；weak-reference：Stage 3 frozen predictions（**仅用于事后评估**）；
- 运行：smoke（5 候选，链路通过）→ run-all（C1/C2/C3 全量，输出 `mhs1_candidates.jsonl` / `mhs1_eventness_bins.jsonl` / `mhs1_split_decisions.jsonl` / `mhs1_summary.json` / `mhs1_evaluation.json`）；
- 输出目录：`/root/autodl-tmp/outputs/stage2_5_mhs1_dev_20260922_134349/`（未上传）。

### 运行过程中的两处工程修复（已提交）

1. 首轮实现为"子段相接式拆分"（不丢弃谷区）→ merged 口径下与 parent 完全相同（指标不动），与任务预期的 coverage 下降矛盾 → 重构为**谷区丢弃式拆分 + pad 仅作用于候选外边界**；
2. smoke 评估暴露公平性问题：limit 模式下 baseline 使用全量视频而 MHS-1 仅覆盖部分视频 → 修复为 **baseline 限定在 MHS-1 实际覆盖的视频集合内**。

## 4. Dev 结果（166 视频 / 393 候选，weak-reference development metrics）

| 配置 | P | R | F1 | tIoU | cov(秒) | ΔP | ΔR | ΔF1 | ΔtIoU | Δcov | split率 | fallback率 | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline（Stage 3 frozen） | 0.5736 | 0.9445 | 0.6665 | 0.5442 | 10.96 | — | — | — | — | — | — | — | — |
| MHS-1-C1 | 0.5729 | 0.9368 | 0.6637 | 0.5404 | 10.85 | **-0.0007** | -0.0076 | **-0.0028** | -0.0038 | -0.108 | 2.3%（9/393） | 0.8% | **FAIL** |
| MHS-1-C2 | 0.5730 | 0.9375 | 0.6640 | 0.5408 | 10.85 | **-0.0007** | -0.0070 | **-0.0025** | -0.0035 | -0.108 | 3.1%（12/393） | 0.3% | **FAIL** |
| MHS-1-C3 | 0.5728 | 0.9384 | 0.6640 | 0.5410 | 10.90 | **-0.0009** | -0.0061 | **-0.0025** | -0.0032 | -0.060 | 1.3%（5/393） | 1.0% | **FAIL** |

失败项均为：`precision_delta_min`（需 ≥+0.005，实际 -0.0007~-0.0009）、`f1_or_tiou_delta_min`（需 ≥+0.003，实际负值）。Recall 均保住（ΔR -0.006~-0.008 > -0.01），coverage 按预期下降，新零 recall = 0。

## 5. 失败机制

1. **可拆分比例极低**：满足"峰距 ≥3-5s + 谷 ≤ 分位阈值 + 段长 ≥3-5s + coverage cap"全部 guard 的候选仅 5-12 个（1.3-3.1%）；可视化中的"多运动峰"大多不满足"峰间有明显低谷"或子段时长门槛（运动起伏 ≠ 事件间隔）；
2. **拆下来的谷区不含"多余内容"**：少数被拆候选丢弃谷区后 precision 不升（-0.0007），说明这些区域的覆盖本来就不是 precision 的主要损失来源 —— precision 损失主要由**整体偏宽的候选覆盖**造成，而不是"大段中间的普通内容"；
3. 因此本轮证明：**在 weak-reference merged 口径下，MHS-1 式拆分不是 precision 提升的有效杠杆**（拆分收益被低触发率 + 谷区无收益双重稀释）。

## 6. 合规性

- 未访问 Heldout（loader 对 heldout split 直接拒绝；仅 dev）；
- 未使用 Hard 调参；未训练 / 未微调；未调用外部 API；
- 未修改 Stage 2/3 正式主线（独立 Stage 2.5 分支）；
- 拆分决策不使用 weak-reference（provenance 全链路声明 `uses_weak_reference_for_decision=false`）；
- 未提交 videos / runs / outputs / 图片 / 模型（仅源码、配置、测试与本报告）。

## 7. 测试与实现

- 新增：`v2/configs/stage2_5/mhs1_formal_splitter.yaml`、`v2/src/video_highlight/stage2_5_multi_highlight_split/`（io/eventness/splitter/schema/evaluate/pipeline）、`v2/scripts/run_mhs1_splitter.py`、`v2/tests/stage2_5/test_mhs1_splitter.py`；
- 定向测试：**15 passed**（eventness schema/finite、单峰 fallback、多峰 split、平谷 fallback、min duration、coverage cap、never-drop-all、schema、合规、CLI）；全仓：**94 passed**。

## 8. 下一步建议

1. **MHS-1 = NEGATIVE_RESULT，不进入 formal integration**；不再继续拆分方向（不强推 val 调参）；
2. 优先转向 **CRR-1 — Stage 2 粗候选 reranking**：问题更可能在"候选级价值排序"（低价值候选拉低 precision），而非段内拆分；
3. 或冻结 temporal/候选形态相关工作，转 Stage 5 spatial / 提交策略；
4. 不建议：继续调 MHS 阈值（本轮已经锁定过一次配置集）、回到 SBC prompt 小修、新增 BR-4。
