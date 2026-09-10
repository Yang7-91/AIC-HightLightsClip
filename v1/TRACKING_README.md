# v1_qwen.py 第二阶段 SAM 2 改造

## 环境

SAM 2 官方当前要求 Python 3.10+、PyTorch 2.5.1+、TorchVision 0.20.1+；Windows 建议使用 WSL Ubuntu。按照官方仓库安装 SAM 2，并下载与配置匹配的 SAM 2.1 checkpoint：

https://github.com/facebookresearch/sam2

推荐起点：

- config: `configs/sam2.1/sam2.1_hiera_s.yaml`
- checkpoint: `sam2.1_hiera_small.pt`

## 运行

```bash
python v1_qwen.py \
  --stage2-backend sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint /path/to/sam2.1_hiera_small.pt \
  --video-dir /path/to/video \
  --out predictions_tracking.jsonl
```

原第二阶段仍可用于消融或兜底：

```bash
python v1_qwen.py --stage2-backend linear --out predictions_linear.jsonl
```

默认在每个高光片段内检测切镜，每个镜头首帧由 Qwen 输出 `subject_box`，SAM 2 逐帧传播掩码。异常轨迹会尝试一次 Qwen + SAM 2 重新初始化；仍失败的帧使用镜头内最近有效框。整个镜头失败时，默认回退到原线性方案，可用 `--tracking-fallback center|error` 修改。

## 重要输出日志

- `tracking_anchor`：镜头初始化主体框及 Qwen 原始输出；
- `tracking_reinit`：漂移后重新初始化；
- `tracking_summary`：镜头范围、修复前后异常帧数。

最终框由主体掩码外接框、边距、离散尺度、EMA 平滑和边界投影生成。无论选择何种 backend，正式提交前都应运行赛事格式和原视频边界校验。
