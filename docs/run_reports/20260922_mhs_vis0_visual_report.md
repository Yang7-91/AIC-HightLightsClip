# MHS-VIS-0 — 多高光候选可视化诊断报告

日期：2026-09-22
结论：**可视化工具链路跑通；Dev 候选普遍存在"一个大段包含多个运动峰"现象（11 个选中候选里 8 个多峰），支持"MHS-1 多高光拆分"问题成立，但本报告仅为诊断，不构成算法结论。**

## 1. 为什么需要可视化

用户观察到一个核心问题：Stage 2 粗高光模型有时不是漏掉高光，而是把多个高光事件合并成一大段
（`[高光A] 普通内容 [高光B] 普通内容 [高光C]` → 一个大 candidate）。在推进 MHS-1（YOLO-guided
Multi-Highlight Splitter）之前，需要一个直观工具回答：①大候选里是否真的有多个高光事件；
②运动/事件曲线能否对应视觉动作峰；③proposed subsegments 是否比原始大段更合理；④哪些视频适合 MHS-1。

## 2. 本轮实现了什么（diagostic-only）

新增 Stage 2.5 可视化诊断分支（**未改 Stage 2/3 主线，未生成任何正式预测**）：

- `v2/configs/stage2_5/mhs_vis0_visual_report.yaml`（冻结配置）
- `v2/src/video_highlight/stage2_5_visual_report/`：`io.py`（多格式只读加载 + Heldout 拒绝）、`selector.py`（确定性示例选择）、`thumbnails.py`（1 fps 缩略图 + contact sheet）、`timeline.py`（逐点运动/镜头切换计算 + 纯 SVG 时间轴）、`summarizer.py`（规则式中文简介）、`html_report.py`（HTML 报告）、`pipeline.py`（编排）
- `v2/scripts/run_mhs_visual_report.py`（`run` / `smoke` / `--video-id` 手动选择）
- `v2/tests/stage2_5/test_mhs_visual_report.py`（15 项测试）

每个候选产出：原始 candidate 灰条 + proposed subsegments 彩条（若存在）+ 运动曲线 + 镜头切换竖线 +
运动峰标记的内联 SVG 时间轴；1 fps 缩略图墙（子片段内帧加绿框）；子片段表格与规则式简介。

## 3. 数据与运行

- 输入：frozen candidate cache（`cache_build_a`，**仅加载 dev split**；heldout 任何形式拒绝加载）+ Dev 166 个视频 + frozen dev manifest
- MHS-1 产物：**不存在** → before-only 模式；eventness 曲线不可用，时间轴以视频自身 **motion proxy**（帧差）代替并在 HTML/报告中明确标注；YOLO density 无数据则省略
- 选择规则：时长长 / 多运动峰 / proposed subsegments>1 / 覆盖率高；`max_videos=6`，每视频最多 2 个候选
- 运行（AutoDL，本地只读）：**smoke 3 候选 / 3 视频**；**full 11 候选 / 6 视频**，共 11 张 contact sheet（127 个 assets）

## 4. 结果概览

- **11 个选中候选里 8 个存在 ≥2 个运动峰**（5 峰 1 个、4 峰 5 个、2 峰 3 个）—— 从运动信号角度支持"一个大段包含多个高光事件"的观察；
- 长候选（12-19s）内部运动峰间隔 3-5s，形态上呈"峰-谷-峰"结构，与"高光A + 普通内容 + 高光B"的假设一致；
- 部分候选（如 qvh_000170、qvh_000319）候选内还有 2-3 次镜头切换，可作为拆分锚点。

### 典型例子（3 个）

| # | video_id | 原 candidate | 运动峰 | 镜头切换 | proposed subsegments | 可视化判断 |
|---|---|---|---|---|---|---|
| 1 | qvh_000005_9x16 | [0.00s, 19.09s]（19.1s，long+multi_peak+high_coverage） | **5 个** @ 0.5s/5.5s/9.5s/14.5s/18.0s | 0 | 0（无 MHS-1） | 强支持多高光合并：5 个明显运动峰贯穿整段 |
| 2 | qvh_000170_9x16 | 17.2s（multi_peak+high_coverage） | **4 个** | **3 次** | 0（无 MHS-1） | 支持：4 峰 + 3 次镜头切换，疑似 3-4 个事件段 |
| 3 | qvh_000254_9x16 | 15.0s（multi_peak+high_coverage） | **4 个** | 0 | 0（无 MHS-1） | 支持：峰谷交替明显，疑似 2-4 个高光段 |

（全部 11 个候选的逐条明细见运行目录 `mhs_vis0_report.md`；HTML 内嵌 SVG 时间轴 + contact sheet 可直观看峰谷结构。）

## 5. 输出位置（本地诊断产物，未上传 GitHub）

- full：`/root/autodl-tmp/outputs/stage2_5_mhs_vis0_report_20260922_130357/`
  - `visual_report.html`（主报告）、`visual_summary.json`、`selected_examples.jsonl`、`mhs_vis0_report.md`、`assets/*.png`（11 张 contact sheet + 116 张缩略图）
- smoke：`/root/autodl-tmp/outputs/stage2_5_mhs_vis0_smoke_20260922_130034/`
- 代码修复记录：加载仅限 dev split（拒绝 heldout）、无视频候选剔除、summary Path 序列化、选择顺序保持评分优先

## 6. MHS-1 / VLM 使用情况

- 是否使用 MHS-1 eventness：**否**（MHS-1 尚未运行；时间轴以 motion proxy 代替并明确标注）
- 是否使用 VLM 简介：**否**（本轮 `optional_vlm: false`，全部为规则式中文简介）

## 7. 合规说明

- 不访问 Heldout（io 层对 heldout split 直接拒绝；仅加载 dev）
- 不修改 predictions / 不生成正式 submission；不做人工修正
- 不向公开平台/第三方服务上传任何视频；不调用外部 API
- 视频 / 图片 / HTML / runs / outputs 均未提交 GitHub（本轮只提交代码、配置、测试与本报告）
- 可视化仅作为诊断与报告用途，不作为人工标注工具

## 8. 测试

- 定向：`python -m unittest discover -s v2/tests/stage2_5` → **15 passed**（selector 确定性 / long 优先 / multi-peak 优先 / timeline schema / SVG 非空 / 规则式简介 / HTML 标记 / heldout 拒绝 / CLI help）
- 全仓：`python -m unittest discover -s v2/tests` → **73 passed**（无既有失败）

## 9. 下一步建议

1. 用 HTML 报告人工观察 11 个典型候选（尤其 qvh_000005 / qvh_000170 / qvh_000254）确认"多峰段"是否对应多个语义高光；
2. 若观察成立：推进 **MHS-1 formal**（YOLO-guided multi-highlight splitter），并让 MHS-1 产出 `mhs1_eventness_bins.jsonl` / `mhs1_proposed_subsegments.jsonl`，本工具会自动叠加 eventness 曲线与 proposed subsegments 对比（before/after）；
3. 若观察不成立（多峰只是同一事件的运动起伏）：不做 MHS-1，转 **Stage 2 粗候选 reranking（CRR-1）**；
4. 不继续 SBC prompt 小修（SBC-0/SBC-1 已闭环负结果）。
