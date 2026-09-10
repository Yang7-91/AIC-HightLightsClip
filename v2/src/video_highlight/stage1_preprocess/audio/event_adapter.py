"""根据基础声学特征生成通用声音变化事件。"""

from __future__ import annotations

from typing import Any

import numpy as np

from video_highlight.contracts.schema_versions import STAGE1_SCHEMA_VERSION


def _robust_z(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-6)
    return ((values - median) / scale).astype(np.float32)


def detect_audio_events(
    video_id: str,
    features: dict[str, np.ndarray],
    z_threshold: float = 2.5,
    onset_db_threshold: float = 6.0,
) -> list[dict[str, Any]]:
    energy = features["rms_dbfs"]
    flux = features["spectral_flux"]
    energy_z = _robust_z(energy)
    flux_z = _robust_z(flux)
    delta = np.diff(energy, prepend=energy[:1]) if len(energy) else energy
    events: list[dict[str, Any]] = []
    for index in range(len(energy)):
        labels: list[tuple[str, float]] = []
        if float(delta[index]) >= onset_db_threshold:
            labels.append(("audio_onset", min(1.0, float(delta[index]) / 18.0)))
        if float(energy_z[index]) >= z_threshold:
            labels.append(("high_energy", min(1.0, float(energy_z[index]) / 6.0)))
        if float(flux_z[index]) >= z_threshold:
            labels.append(("spectral_change", min(1.0, float(flux_z[index]) / 6.0)))
        for label, score in labels:
            events.append(
                {
                    "schema_version": STAGE1_SCHEMA_VERSION,
                    "video_id": video_id,
                    "event_id": len(events),
                    "start_sec": float(features["start_sec"][index]),
                    "end_sec": float(features["end_sec"][index]),
                    "label": label,
                    "score": score,
                    "rms_dbfs": float(energy[index]),
                    "spectral_flux": float(flux[index]),
                }
            )
    return events
