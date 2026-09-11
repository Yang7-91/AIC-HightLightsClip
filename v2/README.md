# Video Highlight Pipeline V2

五阶段视频高光剪辑推理工程。目前已完成 Stage 1：

- FFprobe 读取视频流、音频流、FPS、时间基、尺寸、旋转和时长；
- PySceneDetect `ContentDetector` 完成硬切镜头检测，可选 `ThresholdDetector` 检测渐变；
- OpenCV 顺序解码并按时间戳完成 2 FPS 粗采样；
- 建立抽样帧、原始帧、时间戳和镜头编号映射；
- 按 24 秒窗口、25% 重叠和镜头边界规划 Stage 2 分析片段；
- FFmpeg 保留 16 kHz 连续音轨，生成基础声学特征、事件和时间线；
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

响应中的相对时间由确定性代码映射为原视频绝对时间，重叠窗口候选会扩展、
合并并生成稳定的 `candidate_id`。Base64 本体不会写入请求日志。

## 测试

```powershell
python -m unittest discover -s tests/stage1 -v
python -m unittest discover -s tests/stage2 -v
```
