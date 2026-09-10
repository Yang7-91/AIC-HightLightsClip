"""使用 PySceneDetect 生成镜头区间。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector

from video_highlight.common.exceptions import ArtifactValidationError
from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION


def detect_scenes(
    video_id: str,
    video_path: str | Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """使用内容切换检测，可选叠加渐入渐出检测。"""
    backend = str(config.get("backend", "opencv"))
    video = open_video(str(Path(video_path).resolve()), backend=backend)
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=float(config.get("content_threshold", 27.0)),
            min_scene_len=float(config.get("min_scene_length_sec", 0.75)),
        )
    )
    if bool(config.get("detect_fades", True)):
        manager.add_detector(
            ThresholdDetector(
                threshold=float(config.get("fade_threshold", 12.0)),
                min_scene_len=float(config.get("min_scene_length_sec", 0.75)),
                add_final_scene=True,
            )
        )
    manager.detect_scenes(
        video=video,
        frame_skip=int(config.get("frame_skip", 0)),
        show_progress=bool(config.get("show_progress", False)),
    )
    scene_list = manager.get_scene_list(start_in_scene=True)
    if not scene_list:
        scene_list = [(video.base_timecode, video.duration)]
    scenes: list[dict[str, Any]] = []
    for scene_id, (start, end) in enumerate(scene_list):
        start_sec = float(start.get_seconds())
        end_sec = float(end.get_seconds())
        if end_sec <= start_sec:
            continue
        scenes.append(
            {
                "schema_version": STAGE1_SCHEMA_VERSION,
                "video_id": video_id,
                "scene_id": scene_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": int(start.get_frames()),
                "end_frame": int(end.get_frames()),
                "frame_interval": "[start_frame,end_frame)",
                "transition_type": "pyscenedetect",
                "detectors": ["content", "threshold"] if config.get("detect_fades", True) else ["content"],
            }
        )
    if not scenes:
        raise ArtifactValidationError(f"PySceneDetect 未能生成有效镜头: {video_path}")
    return scenes
