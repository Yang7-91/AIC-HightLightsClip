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

## 测试

```powershell
python -m unittest discover -s tests/stage1 -v
```
