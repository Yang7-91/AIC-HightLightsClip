# Video Highlight Pipeline V2

五阶段视频高光剪辑推理工程。目前已完成 Stage 1 至 Stage 5：

- FFprobe 读取视频流、音频流、FPS、时间基、尺寸、旋转和时长；
- PySceneDetect `ContentDetector` 完成硬切镜头检测，可选 `ThresholdDetector` 检测渐变；
- OpenCV 顺序解码并按时间戳完成 2 FPS 粗采样；
- 建立抽样帧、原始帧、时间戳和镜头编号映射；
- 按 24 秒窗口、25% 重叠和镜头边界规划 Stage 2 分析片段；
- FFmpeg 保留 16 kHz 连续音轨，生成基础声学特征、事件和时间线；
- 生成粗高光候选、细化帧级边界，并完成主体跟踪与平滑构图；
- 按原始索引顺序生成和严格校验最终比赛提交 JSONL；
- 每个视频独立持久化、异常隔离，并支持恢复运行。

## 数据位置

原始数据默认位于代码目录之外：

```text
F:/datasets/video-clip/
├─ test_index.json
└─ video/
   ├─ 0.mp4
   └─ ...
```

代码只读取该目录，不复制或修改原始视频。默认路径定义在 `configs/paths.yaml`。

## 环境

```powershell
conda activate video-clip
python -m pip install "scenedetect==0.7.1" "PyYAML==6.0.3"
```

也可以使用 `environment.yml` 更新环境。Stage 1 依赖 FFmpeg/FFprobe 可执行文件在 PATH 中可用。

## 运行 Stage 1

先处理一个视频：

```powershell
python scripts/run_stage1.py --run-id smoke --video-id 0 --strict
```

处理索引中的全部视频：

```powershell
python scripts/run_stage1.py --run-id baseline_stage1
```

断点续跑：

```powershell
python scripts/run_stage1.py --run-id baseline_stage1 --resume
```

输出位于 `runs/<run-id>/stage1/`。每个视频目录包含：

```text
metadata.json
scenes.jsonl
segments.jsonl
sample_map.jsonl
coarse_frames/*.jpg
audio.wav
audio_features.npz
audio_events.jsonl
audio_timeline.jsonl
asr.jsonl
audio_status.json
_SUCCESS.json
```

当前 ASR 后端默认禁用，`asr.jsonl` 为空；这避免在没有明确 ASR 模型时伪造语音结果。连续音频和声学时间线均已保留，后续可接入本地 ASR。

## Stage 2：Qwen3.5-4B 粗高光候选

Stage 2 通过已经部署好的 vLLM OpenAI 兼容接口调用
`QuantTrio/Qwen3.5-4B-AWQ`，项目中不包含模型加载或 vLLM 部署代码。
服务地址、模型名、API Key、超时、生成参数和视频传输模式均位于
`configs/stage2/qwen3_5_4b.yaml`。

当前默认服务配置为：

```text
base_url: http://172.25.254.120:8000/v1
model: QuantTrio/Qwen3.5-4B-AWQ
api_key: EMPTY
```

运行时必须明确指定某次 Stage 1 产物：

```powershell
python scripts/run_stage2.py `
  --stage1-dir runs/baseline_stage1/stage1 `
  --run-id baseline_stage2 `
  --strict
```

可通过命令行临时覆盖部署位置和模型名称：

```powershell
python scripts/run_stage2.py `
  --stage1-dir runs/baseline_stage1/stage1 `
  --base-url http://127.0.0.1:8000/v1 `
  --model QuantTrio/Qwen3.5-4B-AWQ
```

### 视频传输模式

`data_url` 模式是默认模式。它读取 Stage 1 已经生成的 2 FPS JPEG，不重复
解码原视频；客户端只将这些帧临时编码为低帧率 MP4，然后以 Base64
`data:video/mp4` URL 发送：

```yaml
video_input:
  mode: data_url
```

`url` 模式不在客户端读取或编码媒体，只向 vLLM 发送 URL：

```yaml
video_input:
  mode: url
  url_template: "http://media-server/video/{video_id}?start={start_ms}&end={end_ms}"
```

URL 模板支持 `video_id`、`segment_id`、`start_sec`、`end_sec`、`start_ms`、
`end_ms` 和 `source_name`。推荐 URL 本身返回对应片段。若 URL 指向完整视频，
只有在当前 vLLM 媒体解析器明确支持时，才应开启 `include_time_range_fields`。

Stage 2 每个视频输出：

```text
requests.jsonl
raw_responses.jsonl
analyses_segment_results.jsonl
candidates.jsonl
subject_hints.jsonl
_SUCCESS.json
```

Stage 2 只输出主体文本 `subject`，不输出 `subject_point`。低帧率粗采样上的
点定位误差较大，空间点提示将由后续 Stage 3.5 在帧级高光区间确定后单独生成。

响应中的相对时间由确定性代码映射为原视频绝对时间，重叠窗口候选会扩展、
合并并生成稳定的 `candidate_id`。Base64 本体不会写入请求日志。

## Stage 3：高光候选帧级边界定位

Stage 3 读取同批次的 Stage 1 元数据和 Stage 2 候选。默认只对候选区间进行
10 FPS 精解码，提取运动、直方图变化、清晰度、亮度、音频、镜头边界和
Stage 2 粗分数，通过无训练规则后端保守细化边界，并映射成原始视频的左闭
右开帧区间 `[start_frame, end_frame)`。

正常运行：

```powershell
python scripts/run_stage3.py `
  --stage1-dir runs/full/stage1 `
  --stage2-dir runs/full/stage2 `
  --run-id full_stage3 `
  --strict
```

完全跳过 Stage 3 处理，只把 Stage 2 秒区间映射并封装为合法 Stage 3 输出：

```powershell
python scripts/run_stage3.py `
  --stage1-dir runs/full/stage1 `
  --stage2-dir runs/full/stage2 `
  --run-id full_stage3_passthrough `
  --skip-processing `
  --strict
```

`--skip-processing`（别名 `--passthrough`）不会打开或解码源视频，不提取特征，
也不会执行边界细化、无高光门控或区间合并。Stage 2 的 `start_sec/end_sec`
保持原值，只进行 Stage 4 所需的确定性帧号映射。

每个视频输出：

```text
refined_intervals.jsonl
diagnostics.jsonl
_SUCCESS.json
```

Stage 3 只负责时间边界和主体语义透传，`refined_intervals.jsonl` 不包含
`subject_point`；即使读取旧版 Stage 2 产物中的同名字段也会主动丢弃。

未来获得训练好的 TorchScript TCN 权重后，可用 `--backend tcn --checkpoint ...`
切换到模型推理；默认规则后端不依赖 PyTorch，也不包含训练代码。

## Stage 4：主体构图与轨迹优化

Stage 4 以 Stage 3 的 `refined_intervals.jsonl` 作为高光区间、主体语义和主体点提示的
唯一上游契约。它只从 Stage 1 读取源视频路径、原始帧率/尺寸、`targetRatioWH` 和镜头
边界；命令行没有 Stage 2 参数，也不会读取 Stage 2 产物。

默认使用不需要额外模型权重的 OpenCV 后端：在区间首帧或镜头切换处根据 Stage 3
主体点初始化；没有主体点时使用中心偏置视觉显著性；随后通过稀疏光流逐帧传播主体框。
每帧围绕主体生成多尺度、多偏移、运动方向留白的目标比例候选框，使用动态规划选择
低代价轨迹，再对中心和尺度做限速平滑。镜头边界两侧分别优化，不跨硬切镜头平滑。
所有框最后统一取整、再次限界，并输出比赛需要的 `[x, y, w]`。

正常运行：

```powershell
python scripts/run_stage4.py `
  --stage1-dir runs/full/stage1 `
  --stage3-dir runs/full/stage3 `
  --output-dir runs/full/stage4 `
  --strict
```

可选后端：

- `--backend opencv`：默认可运行基线，不需要新权重。
- `--backend center`：不解码视频的中心最大合法框，用于链路检查或区间失败降级。
- `--backend sam2 --sam2-checkpoint <权重路径> --sam2-config <模型配置>`：使用官方
  SAM2 视频预测器传播主体 Mask；只有选择该后端时才加载 SAM2 和 PyTorch。

每个视频持久化输出：

```text
crops.jsonl         # Stage 5 直接消费的逐帧 [x,y,w]
tracks.jsonl        # 主体 xyxy、置信度和跟踪来源，便于调试
diagnostics.jsonl   # 区间状态、镜头子段数和降级信息
_SUCCESS.json
```

## Stage 5：最终提交 JSONL

Stage 5 只把 Stage 4 的 `crops.jsonl` 作为预测来源。它读取原始 `test_index.json`
以严格保持视频行顺序，并从 Stage 1 元数据复核目标比例、总帧数和画面尺寸；不读取
Stage 2 或 Stage 3。导出时删除 Stage 4 的调试字段，仅保留比赛规定字段。

```powershell
python scripts/run_stage5.py `
  --stage1-dir runs/full/stage1 `
  --stage4-dir runs/full/stage4 `
  --output-dir runs/full/stage5 `
  --overwrite
```

`--input-index` 默认取 `configs/paths.yaml` 中的 `input_index`。最终目录包含：

```text
submission.jsonl       # 可直接提交的最终文件
validation_report.json # 视频数、预测帧数、空结果数、文件 SHA-256
resolved_config.json
run_manifest.json
_SUCCESS.json
```

也可脱离生成流程单独复核一个提交文件：

```powershell
python scripts/validate_submission.py `
  --submission runs/full/stage5/submission.jsonl `
  --input-index F:/datasets/video-clip/test_index.json `
  --stage1-dir runs/full/stage1
```

## 测试

```powershell
python -m unittest discover -s tests/stage1 -v
python -m unittest discover -s tests/stage2 -v
python -m unittest discover -s tests/stage3 -v
python -m unittest discover -s tests/stage4 -v
python -m unittest discover -s tests/stage5 -v
```
