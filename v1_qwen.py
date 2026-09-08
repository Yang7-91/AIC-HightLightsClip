#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (Qwen3.5-4B-awq-int4 two-stage) for the Highlight Video Re-framing task.
基线方案（Qwen3.5-4B-awq-int4 两阶段）：视频高光重构图任务。

本基线仅读取公开的测试索引（video_id + 目标画幅）与输入视频，不依赖任何真值标注。

Two stages 两阶段:
  阶段1 高光定位：把整段视频喂给视频大模型，预测高光时间区间（秒）。

  阶段2 逐帧重构图：Qwen在每个镜头选择重要主体并给出提示框，SAM 2
        在镜头内传播逐帧掩码，再生成动态尺度、平滑且合法的构图框。
        可用 --stage2-backend linear 回到原中心点+线性插值基线。

输出（提交格式，每行一个视频对象）:
    {"video_id": "0", "targetRatioWH": [16, 9],
     "predictions": [{"frame": 10, "bboxes": [x, y, w]}, ...]}
  bboxes 为三元组 [x, y, w]；高度由评测程序按 h = w * th/tw 自动补算。

注：此代码采用的是本地windows环境运行代码+wsl子系统上使用vllm部署模型并暴露出OpenAPI兼容的http接口来进行测试
    为了尽量减少对baseline的修改，所以保留了原有的本地文件检测，但实际使用的数据文件在wsl系统中（未进行文件检测），
    与本地文件完全一致【代码行数定位：350、621】

Dependencies: torch, transformers, qwen-vl-utils, opencv-python, pillow, numpy,openai,vllm
"""
import base64
import io
import os
import re
import json
import argparse
import math

import cv2
import time
from PIL import Image
from openai import OpenAI

from tracking_stage import (
    SAM2VideoTracker,
    detect_shots,
    fill_missing_boxes,
    identify_bad_track_frames,
    plan_smoothed_crops,
)


# --------------------------- IO: test index ---------------------------
def load_index(index_path):
    """Read public test index -> [(video_id, (tw, th)), ...].

    Each item: {"video_id": "0", "targetRatioWH": [16, 9]}.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        vid = str(it["video_id"])
        tr = it.get("targetRatioWH", [16, 9])
        tw, th = float(tr[0]), float(tr[1])
        out.append((vid, (tw, th)))
    return out


def video_meta(video_path):
    """获取视频元信息

    return: n：视频总帧数, fps：视频帧率, w：宽, h：高
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    # 以下对视频帧的处理健壮性欠缺，可进一步优化
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def extract_frames(video_path, frame_ids):
    """Read BGR frames by id -> {frame: ndarray}. Sequential read avoids seeks."""
    frame_ids = sorted(set(int(f) for f in frame_ids))
    out = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or not frame_ids:
        cap.release()
        return out
    start, last = frame_ids[0], frame_ids[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    want = set(frame_ids)
    idx = start
    while idx <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            out[idx] = fr
        idx += 1
    cap.release()
    return out


# --------------- Geometry: crop size + center placement ---------------
def compute_crop_size(W, H, tw, th):
    """Largest target-ratio rectangle that fits inside the source frame."""
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    target = float(tw) / float(th)
    if W / float(H) >= target:        # source wider than target -> full height
        ch = H
        # Width is submitted as an integer; floor keeps the derived height legal.
        cw = min(int(math.floor(H * target)), W)
    else:                              # source taller than target -> full width
        cw = W
        # ceil is only used for clamping y; evaluator derives the exact height
        # from submitted width, so this prevents a fractional-pixel overflow.
        ch = min(int(math.ceil(W / target)), H)
    return max(1, cw), max(1, ch)


def center_to_box(cx, cy, W, H, cw, ch):
    """Normalized center (0~1) -> pixel crop box [x, y, cw, ch], clamped.

    params:
        cx: 裁剪x坐标，已归一化至(0~1)
        cy: 裁剪y坐标，已归一化至(0~1)
        W: 源帧宽度
        H: 源帧高度
        cw: 裁剪宽度
        ch: 裁剪高度
    """
    px = cx * W - cw / 2.0 # 中心点移动至左上角，除以2
    py = cy * H - ch / 2.0
    px = int(round(max(0, min(px, W - cw))))
    py = int(round(max(0, min(py, H - ch))))
    return [px, py, cw, ch]


# --------------------- Parse model outputs ---------------------
_NUM = re.compile(r"-?\d+\.?\d*")


def parse_focus_norm(text, W=None, H=None):
    """Parse subject center, normalize to 0~1. Returns None on failure.

    Robust to coordinate spaces (0~1 float / 0~1000 int / absolute pixel).
    Takes the LAST valid center (the model may reason before answering).
    """
    cand = None
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("center", "subject_center", "focus", "point", "cxcy"):
            v = obj.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    cand = [float(v[0]), float(v[1])]
                except (TypeError, ValueError):
                    pass
    if cand is None:
        nums = _NUM.findall(text)
        if len(nums) >= 2:
            cand = [float(nums[-2]), float(nums[-1])]
    if cand is None:
        return None
    cx, cy = cand

    # TODO 兼容不同的输出格式，进行归一化，有点问题需要修改
    def _norm(v, size):
        if v <= 1.5:
            n = v
        elif v <= 1000.0:
            n = v / 1000.0
        elif size:
            n = v / float(size)
        else:
            n = v / 1000.0
        return max(0.0, min(1.0, n))

    return [_norm(cx, W), _norm(cy, H)]


def parse_subject_box(text, W, H):
    """Parse Qwen subject_box=[x1,y1,x2,y2] into source-pixel coordinates."""
    if not isinstance(text, str):
        return None
    candidate = None
    try:
        objects = [json.loads(text)]
    except Exception:
        objects = []
    for match in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            objects.append(json.loads(match.group(0)))
        except Exception:
            continue
    for obj in objects:
        value = obj.get("subject_box") if isinstance(obj, dict) else None
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                candidate = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    if candidate is None or not all(math.isfinite(item) for item in candidate):
        return None
    x1, y1, x2, y2 = candidate
    # Qwen is instructed to use 0..1000; also tolerate normalized 0..1.
    if max(abs(x1), abs(x2)) <= 1.5 and max(abs(y1), abs(y2)) <= 1.5:
        x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
    elif max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1000.0:
        x1, x2, y1, y2 = x1 * W / 1000.0, x2 * W / 1000.0, y1 * H / 1000.0, y2 * H / 1000.0
    x1, x2 = max(0.0, min(x1, W - 1.0)), max(1.0, min(x2, float(W)))
    y1, y2 = max(0.0, min(y1, H - 1.0)), max(1.0, min(y2, float(H)))
    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None
    return [x1, y1, x2, y2]


def parse_segments_sec(text):
    """Parse highlight intervals [[start_sec, end_sec], ...]. [] on failure."""
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        segs = obj.get("segments")
        if isinstance(segs, list):
            out = []
            for s in segs:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    try:
                        out.append((float(s[0]), float(s[1])))
                    except (TypeError, ValueError):
                        continue
            if out:
                return out
    idx = text.find("segments") # 兜底，纯文本查找方式
    if idx >= 0:
        tail = text[idx:]
        pairs = re.findall(
            r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", tail)
        if pairs:
            return [(float(a), float(b)) for a, b in pairs]
    return []


def sec_segments_to_frames(segs_sec, fps, n_frames):
    """ 将高光以秒分割的区间转换成帧数区间索引
    params:
        segs_sec: 高光时间区间，以秒单位
        fps: 视频帧率
        n_frames: 视频总帧数
    """
    out = []
    for s, e in segs_sec: # 保证每个区间大小、顺序合法
        f0 = int(round(min(s, e) * fps))
        f1 = int(round(max(s, e) * fps))
        f0 = max(0, min(f0, n_frames - 1))
        f1 = max(0, min(f1, n_frames - 1))
        if f1 > f0:
            out.append((f0, f1))
    return out


def merge_segments(segs_frame, n_frames):
    """Clamp to [0, n_frames-1], ensure a<=b, merge overlapping/adjacent.
    params:
        segs_frame: 以帧区间索引的高光，例如[[10,20],[20,40],[50,60]]
        n_frames: 视频总帧数
    return:
        元组数组，如：[(10,40),(50,60)]
    """
    if n_frames <= 0 or not segs_frame:
        return []
    segs = []
    for a, b in segs_frame:
        a, b = int(min(a, b)), int(max(a, b))
        a = max(0, min(a, n_frames - 1))
        b = max(0, min(b, n_frames - 1))
        segs.append([a, b])
    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a <= merged[-1][1] + 1: # 后一个帧区间的开始帧小于等于前一个帧区间的(结尾帧+1)，则合并两个区间
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def lerp_box(b0, b1, t):
    return [int(round(b0[i] + (b1[i] - b0[i]) * t)) for i in range(4)]


def densify_boxes(seg_start, seg_end, key_frames, key_boxes):
    """Per-frame boxes inside a segment by linear interpolation of keyframes."""
    if not key_frames:
        return {}
    pts = sorted(zip(key_frames, key_boxes))
    kfs = [p[0] for p in pts]
    kbs = [p[1] for p in pts]
    out = {}
    for f in range(seg_start, seg_end + 1): # 对于位于[seg_start, seg_end]的中间帧，直接使用右侧关键帧的bbox
        if f <= kfs[0]:
            out[f] = kbs[0]
        elif f >= kfs[-1]:
            out[f] = kbs[-1]
        else:
            j = 0
            while j + 1 < len(kfs) and kfs[j + 1] < f:
                j += 1
            f0, f1 = kfs[j], kfs[j + 1]
            t = (f - f0) / float(f1 - f0) if f1 > f0 else 0.0
            out[f] = lerp_box(kbs[j], kbs[j + 1], t)
    return out


# # --------------------------- Model wrapper ---------------------------
class QwenVL:
    def __init__(self, model_path, device_map="auto", dtype="auto",
                 min_pixels=None, max_pixels=None, enable_thinking=False):
        self.enable_thinking = enable_thinking
        openai_api_key = "EMPTY"
        openai_api_base = "http://172.25.254.120:8000/v1/"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.client = client
        self.model = "QuantTrio/Qwen3.5-4B-AWQ"

    def _generate(self, messages, max_new_tokens=2560):
        pass


    def detect_highlights(self, video_path, fps_sample=2.0, max_new_tokens=2560):
        prompt = (
            "This is a short video. Find the single most highlight-worthy clip "
            "that is worth keeping and re-framing, and give its time interval "
            "in seconds (float, counted from the start of the video).\n"
            "Output ONLY JSON format text, do not use markdown,do not output ```json,no extra text or symbol strictly, example: "
            "{\"segments\": [[start_sec, end_sec]]}"
        )
        video_url = "file:////home/putik-ubuntu/datasets/video-clip/highlight-clip-source/video/" + os.path.basename(video_path)  # 使用本地的文件，减少格式转换，加快数据传输
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]
        hight_light_schema = {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    }
                }
            },
            "required": [
                "segments"
            ]
        }
        ## Use video url in the payload
        chat_completion_from_url = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_completion_tokens=max_new_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "video_segments",
                    "schema": hight_light_schema
                }
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_token_budget": 8000
            }
        )

        raw = chat_completion_from_url.choices[0].message
        if raw is None:
            return []
        raw = {"content":chat_completion_from_url.choices[0].message.content,"reasoning":chat_completion_from_url.choices[0].message.reasoning}
        return parse_segments_sec(raw["content"]), raw

    def predict_focus(self, pil_img, target_ratio, max_new_tokens=128):
        tw, th = target_ratio
        prompt = (
            "Below is a video frame to be re-framed (cropped) to %d:%d.\n"
            "Point out the center of the most important subject / region to "
            "keep. Use normalized integer coordinates in range 0~1000: x is "
            "horizontal (0=left, 1000=right), y is vertical (0=top, "
            "1000=bottom).\n"
            "Output ONLY JSON format text, do not use markdown,do not output ```json,no extra text or symbol strictly, example: {\"center\": [x, y]}"
            % (int(tw), int(th))
        )
        def pil_to_data_url(pil):
            buffer = io.BytesIO()

            pil.save(
                buffer,
                format="JPEG",
                quality=90 # 可控的压缩质量
            )

            base64_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            return f"data:image/jpeg;base64,{base64_image}"

        image_url = pil_to_data_url(pil_img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        focus_point_schema = {
            "type": "object",
            "properties": {
                "center": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"}
                }
            },
            "required": ["center"],
            "additionalProperties": False,
        }
        ## Use video url in the payload
        chat_completion_from_url = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_completion_tokens=max_new_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "focus",
                    "schema": focus_point_schema
                }
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "thinking_token_budget": 4000
            }
        )

        raw = {"content":chat_completion_from_url.choices[0].message.content,"reasoning":chat_completion_from_url.choices[0].message.reasoning}
        return raw # 未做健壮性检查

    def predict_subject_box(self, pil_img, target_ratio, max_new_tokens=256):
        """Select the semantic subject and return a box for SAM 2 prompting."""
        tw, th = map(int, target_ratio)
        prompt = (
            "This frame belongs to a highlight clip that will be reframed to %d:%d. "
            "Select the single most important visible subject or the compact group "
            "that must remain in the crop. Return its tight bounding box using "
            "normalized integer coordinates 0~1000 as [x1,y1,x2,y2]. "
            "Output JSON only: {\"subject_box\":[x1,y1,x2,y2],"
            "\"subject\":\"short description\"}." % (tw, th)
        )
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
        schema = {
            "type": "object",
            "properties": {
                "subject_box": {
                    "type": "array", "minItems": 4, "maxItems": 4,
                    "items": {"type": "number"},
                },
                "subject": {"type": "string"},
            },
            "required": ["subject_box", "subject"],
            "additionalProperties": False,
        }
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            model=self.model,
            max_completion_tokens=max_new_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "tracking_subject", "schema": schema,
            }},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message
        return {"content": message.content, "reasoning": getattr(message, "reasoning", None)}

# --------------------------- Segment processing ---------------------------
def crop_keyframes(model, video_path, seg, target_ratio, stride, W, H, cw, ch,
                   raw_log, vid, max_new_tokens=128):
    """Sample keyframes inside a segment and predict centers -> (frames, boxes)."""
    s, e = seg
    kfs = list(range(s, e + 1, max(1, stride)))
    if kfs[-1] != e:
        kfs.append(e)
    frames_map = extract_frames(video_path, kfs)
    key_frames, key_boxes = [], []
    for f in kfs:
        fr = frames_map.get(f)
        if fr is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        try:
            raw = model.predict_focus(pil, target_ratio, max_new_tokens=max_new_tokens)
            if raw_log is not None:
                raw_log.write(json.dumps(
                    {"video_id": vid, "frame": f, "stage": "crop", "raw": raw},
                    ensure_ascii=False) + "\n")
            content = raw.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型返回的 content 为空或不是字符串")

            fc = parse_focus_norm(content, W, H)
            if fc:
                key_frames.append(f)
                key_boxes.append(center_to_box(fc[0], fc[1], W, H, cw, ch))
        except Exception as ex:
            print("    [crop failed] %s#%d -> %s" % (vid, f, ex))
    if not key_frames:  # fallback: center placement keeps the segment non-empty。此时降级为最原始得兜底策略，所有帧直接取图像中心点为裁剪中心
        key_frames = [s, e]
        cb = center_to_box(0.5, 0.5, W, H, cw, ch)
        key_boxes = [cb, cb]
    return key_frames, key_boxes


def _linear_fallback_predictions(model, video_path, seg, target_ratio, stride,
                                 W, H, cw, ch, raw_log, vid):
    kfs, kbs = crop_keyframes(
        model, video_path, seg, target_ratio, stride, W, H, cw, ch,
        raw_log, vid, max_new_tokens=128)
    dense = densify_boxes(seg[0], seg[1], kfs, kbs)
    return [{"frame": int(frame),
             "bboxes": [int(dense[frame][0]), int(dense[frame][1]), int(dense[frame][2])]}
            for frame in range(seg[0], seg[1] + 1) if frame in dense]


def track_segment(model, tracker, video_path, seg, target_ratio, W, H,
                  raw_log, vid, args, cw, ch):
    """Qwen box initialization + shot-local SAM 2 propagation + crop planning."""
    shots = detect_shots(video_path, seg, args.scene_threshold, args.scene_min_frames)
    predictions = []
    scales = tuple(float(item) for item in args.crop_scales.split(",") if item.strip())
    margins = (args.margin_left, args.margin_right, args.margin_top, args.margin_bottom)
    for shot_start, shot_end in shots:
        try:
            anchor_frame = extract_frames(video_path, [shot_start]).get(shot_start)
            if anchor_frame is None:
                raise RuntimeError("cannot decode shot anchor")
            pil = Image.fromarray(cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2RGB))
            raw = model.predict_subject_box(pil, target_ratio, args.subject_max_tokens)
            content = raw.get("content") if isinstance(raw, dict) else None
            anchor_box = parse_subject_box(content, W, H)
            if anchor_box is None:
                raise ValueError("Qwen subject_box is missing or invalid")
            raw_log.write(json.dumps({
                "video_id": vid, "frame": shot_start, "stage": "tracking_anchor",
                "shot": [shot_start, shot_end], "subject_box": anchor_box, "raw": raw,
            }, ensure_ascii=False) + "\n")

            boxes = tracker.track_range(
                video_path, shot_start, shot_end, anchor_box, (W, H))
            bad_before = identify_bad_track_frames(
                boxes, shot_start, shot_end, (W, H),
                args.min_mask_area_ratio, args.max_mask_area_ratio,
                args.max_area_change, args.max_center_jump)

            # A single recovery pass starts at the first unreliable frame. It
            # creates an independent SAM 2 state, so bad memory is not reused.
            reinitialized_at = None
            if bad_before and args.tracking_reinit > 0:
                reinitialized_at = bad_before[0]
                recovery_frame = extract_frames(video_path, [reinitialized_at]).get(reinitialized_at)
                if recovery_frame is not None:
                    recovery_pil = Image.fromarray(cv2.cvtColor(recovery_frame, cv2.COLOR_BGR2RGB))
                    recovery_raw = model.predict_subject_box(
                        recovery_pil, target_ratio, args.subject_max_tokens)
                    recovery_content = recovery_raw.get("content") if isinstance(recovery_raw, dict) else None
                    recovery_box = parse_subject_box(recovery_content, W, H)
                    if recovery_box is not None:
                        boxes.update(tracker.track_range(
                            video_path, reinitialized_at, shot_end,
                            recovery_box, (W, H)))
                        raw_log.write(json.dumps({
                            "video_id": vid, "frame": reinitialized_at,
                            "stage": "tracking_reinit", "subject_box": recovery_box,
                            "raw": recovery_raw,
                        }, ensure_ascii=False) + "\n")

            bad_after = identify_bad_track_frames(
                boxes, shot_start, shot_end, (W, H),
                args.min_mask_area_ratio, args.max_mask_area_ratio,
                args.max_area_change, args.max_center_jump)
            # Invalid frames are treated as missing before nearest-valid fill.
            for frame in bad_after:
                boxes[frame] = None
            boxes = fill_missing_boxes(boxes, shot_start, shot_end, anchor_box)
            crops = plan_smoothed_crops(
                boxes, shot_start, shot_end, (W, H), target_ratio,
                scales=scales, margins=margins,
                center_alpha=args.center_alpha, width_alpha=args.width_alpha)
            for frame in range(shot_start, shot_end + 1):
                x, y, width, _ = crops[frame]
                predictions.append({"frame": int(frame),
                                    "bboxes": [int(x), int(y), int(width)]})
            raw_log.write(json.dumps({
                "video_id": vid, "stage": "tracking_summary",
                "shot": [shot_start, shot_end], "bad_before": len(bad_before),
                "bad_after": len(bad_after), "reinitialized_at": reinitialized_at,
            }, ensure_ascii=False) + "\n")
        except Exception as exc:
            print("    [tracking failed] %s#%d-%d -> %s" %
                  (vid, shot_start, shot_end, exc))
            if args.tracking_fallback == "error":
                raise
            if args.tracking_fallback == "linear":
                predictions.extend(_linear_fallback_predictions(
                    model, video_path, (shot_start, shot_end), target_ratio,
                    args.crop_stride, W, H, cw, ch, raw_log, vid))
            else:
                center = center_to_box(0.5, 0.5, W, H, cw, ch)
                predictions.extend({"frame": frame,
                                    "bboxes": center[:3]}
                                   for frame in range(shot_start, shot_end + 1))
    return predictions

# --------------------------- Main ---------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    there = os.path.dirname("F:\\baiduNetDisk\\baiduNetDiskDownload\\基于视频大模型的通用视频高光剪辑\\基于视频大模型的通用视频高光剪辑\\") #windows格式，绝对路径指明本地文件
    ap = argparse.ArgumentParser(
        description="Qwen-VL two-stage baseline: highlight localization + re-framing")
    ap.add_argument("--model", required=False, help="local VL model weights path")
    ap.add_argument("--index", default=os.path.join(here, "test_index.json"),
                    help="public test index (video_id + targetRatioWH)")
    ap.add_argument("--video-dir", default=os.path.join(there, "video"))
    ap.add_argument("--out", default=os.path.join(here, "predictions.jsonl"))
    ap.add_argument("--num-videos", type=int, default=0,
                    help="process first N videos, 0 = all")
    ap.add_argument("--crop-stride", type=int, default=15,
                    help="stage-2 keyframe stride (frames)")
    ap.add_argument("--detect-fps", type=float, default=2.0,
                    help="stage-1 sampling fps fed to the model")
    ap.add_argument("--max-new-tokens", type=int, default=9600)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--stage2-backend", choices=("sam2", "linear"), default="sam2",
                    help="sam2: Qwen box + video tracking; linear: original baseline")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml",
                    help="SAM 2 Hydra model config")
    ap.add_argument("--sam2-checkpoint", default=None,
                    help="local SAM 2.1 checkpoint path (required for sam2 backend)")
    ap.add_argument("--sam2-device", default="cuda")
    ap.add_argument("--sam2-amp-dtype", choices=("bfloat16", "float16", "none"),
                    default="bfloat16")
    ap.add_argument("--sam2-vos-optimized", action="store_true")
    ap.add_argument("--scene-threshold", type=float, default=27.0)
    ap.add_argument("--scene-min-frames", type=int, default=15)
    ap.add_argument("--subject-max-tokens", type=int, default=256)
    ap.add_argument("--tracking-reinit", type=int, choices=(0, 1), default=1,
                    help="one Qwen/SAM2 reinitialization after detected drift")
    ap.add_argument("--tracking-fallback", choices=("linear", "center", "error"),
                    default="linear")
    ap.add_argument("--min-mask-area-ratio", type=float, default=0.0005)
    ap.add_argument("--max-mask-area-ratio", type=float, default=0.70)
    ap.add_argument("--max-area-change", type=float, default=4.0)
    ap.add_argument("--max-center-jump", type=float, default=0.20)
    ap.add_argument("--crop-scales", default="0.55,0.70,0.85,1.00")
    ap.add_argument("--margin-left", type=float, default=0.20)
    ap.add_argument("--margin-right", type=float, default=0.20)
    ap.add_argument("--margin-top", type=float, default=0.15)
    ap.add_argument("--margin-bottom", type=float, default=0.30)
    ap.add_argument("--center-alpha", type=float, default=0.25)
    ap.add_argument("--width-alpha", type=float, default=0.15)
    args = ap.parse_args()

    if args.scene_min_frames < 1:
        ap.error("--scene-min-frames must be >= 1")
    if not 0 < args.center_alpha <= 1 or not 0 < args.width_alpha <= 1:
        ap.error("smoothing alphas must be in (0,1]")
    try:
        crop_scales = [float(item) for item in args.crop_scales.split(",") if item.strip()]
    except ValueError:
        ap.error("--crop-scales must be comma-separated numbers")
    if not crop_scales or any(item <= 0 or item > 1 for item in crop_scales):
        ap.error("--crop-scales values must be in (0,1]")
    if any(value < 0 for value in (args.margin_left, args.margin_right,
                                    args.margin_top, args.margin_bottom)):
        ap.error("crop margins must be non-negative")

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[:args.num_videos]
    # index = [("0",(16,9))]
    print("To infer: %d videos" % len(index))

    model = QwenVL(args.model, device_map=args.device_map, dtype=args.dtype,
                   min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                   enable_thinking=args.enable_thinking)
    tracker = None
    if args.stage2_backend == "sam2":
        tracker = SAM2VideoTracker(
            args.sam2_config, args.sam2_checkpoint, args.sam2_device,
            args.sam2_amp_dtype, args.sam2_vos_optimized)

    raw_path = args.out + ".raw.jsonl"
    n_lines = 0
    with open(args.out, "w", encoding="utf-8") as fout, \
            open(raw_path, "w", encoding="utf-8") as raw_log:
        for vi, (vid, target_ratio) in enumerate(index, 1):
            video_path = os.path.join(args.video_dir, vid + ".mp4")
            if not os.path.exists(video_path):
                print("  [skip] no video: %s" % video_path)
                fout.write(json.dumps(
                    {"video_id": vid, "targetRatioWH": [int(target_ratio[0]),
                     int(target_ratio[1])], "predictions": []},
                    ensure_ascii=False) + "\n")
                continue
            n_frames, fps, W, H = video_meta(video_path)
            cw, ch = compute_crop_size(W, H, target_ratio[0], target_ratio[1])

            # Stage 1: highlight localization (model only).
            segs_sec, raw = model.detect_highlights(
                video_path, fps_sample=args.detect_fps,
                max_new_tokens=args.max_new_tokens)
            raw_log.write(json.dumps(
                {"video_id": vid, "stage": "detect", "raw": raw},
                ensure_ascii=False) + "\n")
            segments = merge_segments(
                sec_segments_to_frames(segs_sec, fps, n_frames),
                n_frames
            )

            print("  [%d/%d] %s %dx%d fps=%.2f frames=%d crop=%dx%d segs=%s"
                  % (vi, len(index), vid, W, H, fps, n_frames, cw, ch, segments))

            # Stage 2: per-frame re-framing inside each segment.
            predictions = []
            for seg in segments:
                if args.stage2_backend == "sam2":
                    predictions.extend(track_segment(
                        model, tracker, video_path, seg, target_ratio,
                        W, H, raw_log, vid, args, cw, ch))
                else:
                    predictions.extend(_linear_fallback_predictions(
                        model, video_path, seg, target_ratio,
                        args.crop_stride, W, H, cw, ch, raw_log, vid))
            predictions.sort(key=lambda r: r["frame"])
            rec = {"video_id": vid,
                   "targetRatioWH": [int(target_ratio[0]), int(target_ratio[1])],
                   "predictions": predictions}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_lines += len(predictions)
            fout.flush()
            raw_log.flush()
    print("Done: %d videos / %d frame predictions -> %s"
          % (len(index), n_lines, args.out))
    print("Raw model outputs -> %s" % raw_path)


if __name__ == "__main__":
    start = time.time()
    print(start)
    main()
    end = time.time()
    print(end)
    print(end - start)
