#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 tracking utilities: shot split, SAM 2 propagation and crop planning.

Heavy dependencies (OpenCV, torch and SAM 2) are imported lazily so geometry
and validation code can be unit-tested without loading a model.
"""

from contextlib import ExitStack, nullcontext
import math
from pathlib import Path
import tempfile

import numpy as np


def detect_shots(video_path, segment, threshold=27.0, min_scene_frames=15):
    """Return inclusive shot ranges inside ``segment`` using frame differences."""
    import cv2

    start, end = map(int, segment)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video for scene detection: %s" % video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cuts = []
    previous = None
    last_cut = start
    frame_id = start
    try:
        while frame_id <= end:
            ok, frame = cap.read()
            if not ok:
                break
            thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            current = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV).astype(np.float32)
            if previous is not None:
                # HSV channels have different ranges; normalize to 0..255 first.
                delta = np.abs(current - previous)
                # Hue wraps at 180 in OpenCV; circular distance avoids a false
                # cut when a red hue moves across the 0/179 boundary.
                delta[..., 0] = np.minimum(delta[..., 0], 180.0 - delta[..., 0])
                delta[..., 0] *= 255.0 / 90.0
                score = float(delta.mean())
                if score >= threshold and frame_id - last_cut >= min_scene_frames:
                    cuts.append(frame_id)
                    last_cut = frame_id
            previous = current
            frame_id += 1
    finally:
        cap.release()

    ranges = []
    cursor = start
    for cut in cuts:
        if cut > cursor:
            ranges.append((cursor, cut - 1))
            cursor = cut
    ranges.append((cursor, end))
    return ranges


def largest_component_box(mask):
    """Return [x1,y1,x2,y2) for the largest connected mask component."""
    import cv2

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[index, cv2.CC_STAT_LEFT])
    y = int(stats[index, cv2.CC_STAT_TOP])
    w = int(stats[index, cv2.CC_STAT_WIDTH])
    h = int(stats[index, cv2.CC_STAT_HEIGHT])
    return [x, y, x + w, y + h]


def _decode_range_to_jpegs(video_path, start, end, output_dir, quality=95):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video for SAM 2: %s" % video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    written = 0
    try:
        for global_frame in range(int(start), int(end) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            path = Path(output_dir) / ("%06d.jpg" % written)
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError("failed to write SAM 2 temporary frame: %s" % path)
            written += 1
    finally:
        cap.release()
    expected = int(end) - int(start) + 1
    if written != expected:
        raise RuntimeError("decoded %d/%d frames for range [%d,%d]" %
                           (written, expected, start, end))


class SAM2VideoTracker:
    """Thin adapter around the official SAM 2 video predictor."""

    def __init__(self, model_config, checkpoint, device="cuda",
                 amp_dtype="bfloat16", vos_optimized=False):
        if not model_config or not checkpoint:
            raise ValueError("SAM 2 requires --sam2-config and --sam2-checkpoint")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("SAM 2 checkpoint not found: %s" % checkpoint)
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 is not installed. Install official facebookresearch/sam2 "
                "and a compatible PyTorch build first.") from exc
        self.torch = torch
        self.device = device
        self.amp_dtype = amp_dtype
        kwargs = {"device": device}
        if vos_optimized:
            kwargs["vos_optimized"] = True
        try:
            self.predictor = build_sam2_video_predictor(
                model_config, str(checkpoint_path), **kwargs)
        except TypeError:
            # Older pinned SAM 2 versions do not expose vos_optimized.
            kwargs.pop("vos_optimized", None)
            self.predictor = build_sam2_video_predictor(
                model_config, str(checkpoint_path), **kwargs)

    def _contexts(self):
        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        if str(self.device).startswith("cuda") and self.amp_dtype != "none":
            dtype = getattr(self.torch, self.amp_dtype)
            stack.enter_context(self.torch.autocast("cuda", dtype=dtype))
        else:
            stack.enter_context(nullcontext())
        return stack

    @staticmethod
    def _mask_for_object(object_ids, mask_logits, object_id=1):
        ids = object_ids.tolist() if hasattr(object_ids, "tolist") else list(object_ids)
        ids = [int(x) for x in ids]
        if object_id not in ids:
            return None
        logits = mask_logits[ids.index(object_id)]
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        return np.asarray(logits).squeeze() > 0.0

    def track_range(self, video_path, start, end, anchor_box, frame_size):
        """Track one object from the first frame; return global frame -> xyxy box."""
        import cv2

        width, height = map(int, frame_size)
        if end < start:
            return {}
        with tempfile.TemporaryDirectory(prefix="sam2-shot-") as directory:
            _decode_range_to_jpegs(video_path, start, end, directory)
            results = {}
            state = None
            with self._contexts():
                state = self.predictor.init_state(video_path=directory)
                if hasattr(self.predictor, "reset_state"):
                    self.predictor.reset_state(state)
                _, object_ids, logits = self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(anchor_box, dtype=np.float32),
                )
                first_mask = self._mask_for_object(object_ids, logits)
                if first_mask is not None:
                    if first_mask.shape != (height, width):
                        first_mask = cv2.resize(first_mask.astype(np.uint8),
                                                (width, height),
                                                interpolation=cv2.INTER_NEAREST) > 0
                    results[int(start)] = largest_component_box(first_mask)
                for local_frame, object_ids, logits in self.predictor.propagate_in_video(state):
                    mask = self._mask_for_object(object_ids, logits)
                    if mask is not None and mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                          interpolation=cv2.INTER_NEAREST) > 0
                    results[int(start) + int(local_frame)] = (
                        largest_component_box(mask) if mask is not None else None)
            if state is not None and hasattr(self.predictor, "reset_state"):
                self.predictor.reset_state(state)
            return results


def identify_bad_track_frames(boxes, start, end, frame_size,
                              min_area_ratio=0.0005, max_area_ratio=0.70,
                              max_area_change=4.0, max_center_jump=0.20):
    """Detect empty/implausible boxes; thresholds are intentionally configurable."""
    width, height = frame_size
    frame_area = float(width * height)
    diagonal = math.hypot(width, height)
    previous = None
    bad = []
    for frame in range(start, end + 1):
        box = boxes.get(frame)
        if box is None or len(box) != 4:
            bad.append(frame)
            continue
        x1, y1, x2, y2 = map(float, box)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratio = area / frame_area if frame_area else 0.0
        invalid = not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
        invalid = invalid or ratio < min_area_ratio or ratio > max_area_ratio
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if previous is not None:
            prev_area, prev_center = previous
            area_factor = max(area / max(prev_area, 1.0), prev_area / max(area, 1.0))
            jump = math.hypot(center[0] - prev_center[0],
                              center[1] - prev_center[1]) / max(diagonal, 1.0)
            invalid = invalid or area_factor > max_area_change or jump > max_center_jump
        if invalid:
            bad.append(frame)
        else:
            previous = (area, center)
    return bad


def fill_missing_boxes(boxes, start, end, fallback_box):
    """Fill missing tracker frames with nearest valid boxes (no cross-shot use)."""
    result = {}
    last = None
    for frame in range(start, end + 1):
        box = boxes.get(frame)
        if box is not None:
            last = list(map(float, box))
        result[frame] = list(last) if last is not None else None
    next_box = None
    for frame in range(end, start - 1, -1):
        if result[frame] is not None:
            next_box = result[frame]
        elif next_box is not None:
            result[frame] = list(next_box)
    for frame in range(start, end + 1):
        if result[frame] is None:
            result[frame] = list(map(float, fallback_box))
    return result


def _legal_crop_from_state(cx, cy, width, frame_size, target_ratio):
    W, H = map(int, frame_size)
    rw, rh = map(float, target_ratio)
    max_width = max(1, min(W, int(math.floor(H * rw / rh))))
    width = max(1, min(int(math.floor(width)), max_width))
    height = width * rh / rw
    x = int(round(cx - width / 2.0))
    y = int(round(cy - height / 2.0))
    x = max(0, min(x, W - width))
    y = max(0, min(y, int(math.floor(H - height))))
    return [x, y, width, height]


def subject_box_to_crop(box, frame_size, target_ratio,
                        scales=(0.55, 0.70, 0.85, 1.0),
                        margins=(0.20, 0.20, 0.15, 0.30)):
    """Convert subject xyxy to a target-ratio crop using discrete zoom scales."""
    W, H = map(int, frame_size)
    rw, rh = map(float, target_ratio)
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    left, right, top, bottom = margins
    rx1, rx2 = x1 - left * bw, x2 + right * bw
    ry1, ry2 = y1 - top * bh, y2 + bottom * bh
    required_width = max(rx2 - rx1, (ry2 - ry1) * rw / rh)
    max_width = max(1, min(W, int(math.floor(H * rw / rh))))
    candidates = sorted({max(1, min(max_width, int(math.floor(max_width * s))))
                         for s in scales if s > 0})
    if max_width not in candidates:
        candidates.append(max_width)
    crop_width = next((value for value in candidates if value >= required_width), max_width)
    cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
    return _legal_crop_from_state(cx, cy, crop_width, frame_size, target_ratio)


def plan_smoothed_crops(subject_boxes, start, end, frame_size, target_ratio,
                        scales=(0.55, 0.70, 0.85, 1.0),
                        margins=(0.20, 0.20, 0.15, 0.30),
                        center_alpha=0.25, width_alpha=0.15):
    """Plan and EMA-smooth [x,y,w,h] crop boxes inside one shot."""
    raw = {}
    for frame in range(start, end + 1):
        raw[frame] = subject_box_to_crop(
            subject_boxes[frame], frame_size, target_ratio, scales, margins)
    output = {}
    state = None
    for frame in range(start, end + 1):
        x, y, w, h = raw[frame]
        current = (x + w / 2.0, y + h / 2.0, math.log(max(w, 1.0)))
        if state is None:
            state = current
        else:
            state = (
                state[0] + center_alpha * (current[0] - state[0]),
                state[1] + center_alpha * (current[1] - state[1]),
                state[2] + width_alpha * (current[2] - state[2]),
            )
        output[frame] = _legal_crop_from_state(
            state[0], state[1], math.exp(state[2]), frame_size, target_ratio)
    return output
