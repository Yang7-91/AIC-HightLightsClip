# Stage 1A / Stage 1B / Stage 2 分段运行说明

`v1_qwen.py` 支持四种运行模式：

```text
--run-stage all
--run-stage stage1a
--run-stage stage1b
--run-stage stage2
```

## 数据流

```text
test_index.json + 原视频
        │
        ▼
Stage 1A：Qwen thinking 高光检测
        │ stage1a_segments.jsonl
        ▼
Stage 1B：镜头切分 + Qwen 主体锚点
        │ stage1b_anchors.jsonl
        ▼
Stage 2：SAM2 跟踪 + 构图
        │
        ▼
predictions.jsonl
```

Stage 1A 和 Stage 1B 分别生成一个 `.raw.jsonl` 文件，用于保存 Qwen 原始响应和 reasoning。主中间文件只保留后续阶段需要的结构化数据。

## 一次执行全部阶段

```bash
python v1_qwen.py \
  --run-stage all \
  --index /root/video-clip/test_index.json \
  --video-dir /root/autodl-tmp/video-clip-data/video \
  --stage1a-jsonl /root/video-clip/stage1a_segments.jsonl \
  --stage1b-jsonl /root/video-clip/stage1b_anchors.jsonl \
  --out /root/video-clip/predictions.jsonl \
  --enable-thinking \
  --stage2-backend sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint /root/sam2/checkpoints/sam2.1_hiera_small.pt \
  --tracking-reinit 0 \
  --tracking-fallback error
```

`all` 也会真实写出两个中间 JSONL，再从序列化后的记录继续下一阶段，确保单独运行和一次运行的数据边界一致。

## 分别执行

### Stage 1A

```bash
python v1_qwen.py \
  --run-stage stage1a \
  --index /root/video-clip/test_index.json \
  --video-dir /root/autodl-tmp/video-clip-data/video \
  --stage1a-jsonl /root/video-clip/stage1a_segments.jsonl \
  --detect-fps 1.0 \
  --max-new-tokens 512 \
  --enable-thinking
```

Stage 1A 强制要求 `--enable-thinking`。输出同时保存 `segments_sec` 和经过 FPS 换算、合并与边界裁剪后的 `segments_frame`。Stage 2 最终使用帧区间，避免重复换算产生取整差异。

### Stage 1B

```bash
python v1_qwen.py \
  --run-stage stage1b \
  --video-dir /root/autodl-tmp/video-clip-data/video \
  --stage1a-jsonl /root/video-clip/stage1a_segments.jsonl \
  --stage1b-jsonl /root/video-clip/stage1b_anchors.jsonl \
  --subject-max-tokens 256 \
  --tracking-fallback error
```

Stage 1B 不重新检测高光。它读取 Stage 1A 的帧区间，完成镜头切分，并对每个镜头的起始帧调用 Qwen 生成 `subject_box`。

### Stage 2

```bash
python v1_qwen.py \
  --run-stage stage2 \
  --video-dir /root/autodl-tmp/video-clip-data/video \
  --stage1b-jsonl /root/video-clip/stage1b_anchors.jsonl \
  --out /root/video-clip/predictions.jsonl \
  --stage2-backend sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2-checkpoint /root/sam2/checkpoints/sam2.1_hiera_small.pt \
  --tracking-reinit 0 \
  --tracking-fallback error
```

Stage 2 不创建 Qwen 客户端，也不调用高光或视觉大模型。它只读取 Stage 1B JSONL、原视频和 SAM2 权重。

## 重要限制

- 纯 Stage 2 必须使用 `--tracking-reinit 0`。动态重初始化需要在发现漂移后重新调用 Qwen，与“Stage 2 不调用 Qwen”的边界冲突。
- 分段 Stage 2 当前要求 `--stage2-backend sam2`。
- `linear` fallback 需要再次调用 Qwen，因此纯 Stage 2 不允许使用；请选择 `--tracking-fallback error` 或 `center`。
- 每个中间文件带有 `schema_version`。Stage 1B/Stage 2 会拒绝读取错误阶段的 JSONL。
- Stage 1B 和 Stage 2 会核对实际视频的宽、高、FPS、总帧数；视频被替换后必须重新运行上游阶段。
- `--num-videos N` 在每一种模式下都只处理输入记录的前 N 个视频，便于冒烟测试。

## 中间文件示例

Stage 1A：

```json
{"schema_version":"video-clip.stage1a.v1","video_id":"0","video_name":"0.mp4","targetRatioWH":[16,9],"video_meta":{"width":720,"height":1280,"fps":59.94005994,"num_frames":630},"segments_sec":[[0.0,5.5]],"segments_frame":[[0,330]],"stage1a_config":{"detect_fps":1.0,"max_new_tokens":512,"enable_thinking":true},"status":"ok"}
```

Stage 1B：

```json
{"schema_version":"video-clip.stage1b.v1","video_id":"0","video_name":"0.mp4","targetRatioWH":[16,9],"video_meta":{"width":720,"height":1280,"fps":59.94005994,"num_frames":630},"segments_sec":[[0.0,5.5]],"segments_frame":[[0,330]],"shots":[{"segment_frame":[0,330],"shot_frame":[0,330],"anchor_frame":0,"subject_box":[120.0,180.0,600.0,1180.0],"anchor_status":"ok"}],"status":"ok"}
```

