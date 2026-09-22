# MHS-VIS-1 — Markdown-first 可视化案例报告（脱敏摘要）

日期：2026-09-22
类型：diagnostic-only / 本地报告摘要（**不含视频帧图片**）

## Full 报告（仅本地）

- 本地 full Markdown：`/root/autodl-tmp/outputs/stage2_5_mhs_vis0_report_20260922_130357/mhs_vis0_visual_case_report.md`（含内联 SVG 时间轴与 contact sheet 相对路径图片）
- 本地输出目录：`/root/autodl-tmp/outputs/stage2_5_mhs_vis0_report_20260922_130357`（assets 缩略图与 contact sheet 仅存本地，未上传）

## 统计

- 选中候选数量：11；视频数量：6
- 多峰候选数量（≥2 运动峰）：8
- 典型 video_id：qvh_000005_9x16, qvh_000170_9x16, qvh_000254_9x16

## 结论

- 本轮共选择 11 个候选用于可视化诊断，其中 8 个候选存在多个运动峰（疑似多高光合并）。所有判断均为 diagnostic-only，不用于人工修正预测。
- 支持继续评估 MHS-1 formal（多峰结构在 Dev 候选中普遍存在），但需人工确认多峰是否对应多个语义高光。

## 合规说明

- 未访问 Heldout；未运行 Hard；未训练模型；未调用外部 API；
- 未修改 Stage 2/3 主线；未生成正式 predictions/submission；
- 视频、缩略图、contact sheet、HTML、zip 均未提交 GitHub（仅源码与本脱敏摘要）。

## 下一步建议

1. 下载本地 `$VIS_DIR`（或 zip bundle）查看图文案例；
2. 确认多峰语义后推进 MHS-1 formal（eventness/subsegments 将自动叠加进本报告）；
3. 否则转 Stage 2 coarse reranking（CRR-1），不再做边界/prompt 微调。
