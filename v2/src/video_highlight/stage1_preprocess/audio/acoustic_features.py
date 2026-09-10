"""从单声道 PCM WAV 计算低成本连续声学特征。"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def extract_acoustic_features(wav_path: str | Path, bin_sec: float = 0.5) -> dict[str, np.ndarray]:
    with wave.open(str(wav_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError("声学特征提取只接受单声道 16-bit PCM WAV")
    waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    bin_samples = max(1, int(round(sample_rate * bin_sec)))
    count = math.ceil(len(waveform) / bin_samples) if len(waveform) else 0
    starts: list[float] = []
    ends: list[float] = []
    rms_dbfs: list[float] = []
    peaks: list[float] = []
    zero_crossing: list[float] = []
    fluxes: list[float] = []
    previous_spectrum: np.ndarray | None = None
    for index in range(count):
        start_sample = index * bin_samples
        end_sample = min(len(waveform), start_sample + bin_samples)
        chunk = waveform[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))) if len(chunk) else 0.0
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        zcr = float(np.mean(np.signbit(chunk[:-1]) != np.signbit(chunk[1:]))) if len(chunk) > 1 else 0.0
        windowed = chunk * np.hanning(len(chunk)).astype(np.float32) if len(chunk) else chunk
        spectrum = np.abs(np.fft.rfft(windowed, n=bin_samples)).astype(np.float32)
        total = float(spectrum.sum())
        if total > 0:
            spectrum /= total
        flux = 0.0 if previous_spectrum is None else float(np.maximum(spectrum - previous_spectrum, 0).sum())
        previous_spectrum = spectrum
        starts.append(start_sample / sample_rate)
        ends.append(end_sample / sample_rate)
        rms_dbfs.append(20.0 * math.log10(max(rms, 1e-8)))
        peaks.append(peak)
        zero_crossing.append(zcr)
        fluxes.append(flux)
    return {
        "start_sec": np.asarray(starts, dtype=np.float32),
        "end_sec": np.asarray(ends, dtype=np.float32),
        "rms_dbfs": np.asarray(rms_dbfs, dtype=np.float32),
        "peak": np.asarray(peaks, dtype=np.float32),
        "zero_crossing_rate": np.asarray(zero_crossing, dtype=np.float32),
        "spectral_flux": np.asarray(fluxes, dtype=np.float32),
        "sample_rate": np.asarray([sample_rate], dtype=np.int32),
        "bin_sec": np.asarray([bin_sec], dtype=np.float32),
    }


def feature_rows(features: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    count = len(features["start_sec"])
    return [
        {
            "bin_id": index,
            "start_sec": float(features["start_sec"][index]),
            "end_sec": float(features["end_sec"][index]),
            "rms_dbfs": float(features["rms_dbfs"][index]),
            "peak": float(features["peak"][index]),
            "zero_crossing_rate": float(features["zero_crossing_rate"][index]),
            "spectral_flux": float(features["spectral_flux"][index]),
        }
        for index in range(count)
    ]
