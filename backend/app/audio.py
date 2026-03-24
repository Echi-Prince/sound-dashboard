from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass


@dataclass
class DecodedAudio:
    samples: list[float]
    sample_rate_hz: int
    num_channels: int
    sample_width_bytes: int
    frame_count: int

    @property
    def duration_ms(self) -> int:
        if self.sample_rate_hz == 0:
            return 0
        return int((self.frame_count / self.sample_rate_hz) * 1000)


@dataclass
class ComputedFeatures:
    rms: float
    peak_amplitude: float
    zero_crossing_rate: float
    dominant_activity_ratio: float


def decode_wav(file_bytes: bytes) -> DecodedAudio:
    with wave.open(io.BytesIO(file_bytes), "rb") as wav_file:
        sample_rate_hz = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width_bytes = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()

        if sample_width_bytes not in (1, 2, 4):
            raise ValueError("Only 8-bit, 16-bit, and 32-bit PCM WAV files are supported.")

        raw_frames = wav_file.readframes(frame_count)

    samples = _pcm_to_mono_samples(
        raw_frames=raw_frames,
        num_channels=num_channels,
        sample_width_bytes=sample_width_bytes,
    )

    return DecodedAudio(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
        sample_width_bytes=sample_width_bytes,
        frame_count=frame_count,
    )


def extract_features(samples: list[float], sample_rate_hz: int) -> ComputedFeatures:
    if not samples:
        return ComputedFeatures(
            rms=0.0,
            peak_amplitude=0.0,
            zero_crossing_rate=0.0,
            dominant_activity_ratio=0.0,
        )

    peak_amplitude = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    zero_crossings = 0
    for previous_sample, current_sample in zip(samples, samples[1:]):
        if (previous_sample < 0.0 and current_sample >= 0.0) or (
            previous_sample >= 0.0 and current_sample < 0.0
        ):
            zero_crossings += 1

    zero_crossing_rate = zero_crossings / max(1, len(samples) - 1)

    window_size = max(1, int(sample_rate_hz * 0.05))
    activity_threshold = max(0.02, rms * 0.75)
    active_windows = 0
    total_windows = 0

    for start in range(0, len(samples), window_size):
        window = samples[start : start + window_size]
        if not window:
            continue

        total_windows += 1
        window_rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        if window_rms >= activity_threshold:
            active_windows += 1

    dominant_activity_ratio = active_windows / max(1, total_windows)

    return ComputedFeatures(
        rms=round(rms, 6),
        peak_amplitude=round(peak_amplitude, 6),
        zero_crossing_rate=round(zero_crossing_rate, 6),
        dominant_activity_ratio=round(dominant_activity_ratio, 6),
    )


def build_prototype_detections(
    features: ComputedFeatures, duration_ms: int
) -> list[dict[str, float | int | str]]:
    detections: list[dict[str, float | int | str]] = []
    end_ms = max(duration_ms, 1)

    if features.dominant_activity_ratio >= 0.2 and features.rms >= 0.01:
        label = "speech" if features.zero_crossing_rate >= 0.06 else "traffic"
        confidence = min(
            0.95,
            0.45 + (features.dominant_activity_ratio * 0.4) + (features.rms * 2.5),
        )
        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 3),
                "start_ms": 0,
                "end_ms": end_ms,
            }
        )

    if features.peak_amplitude >= 0.8:
        detections.append(
            {
                "label": "siren",
                "confidence": round(min(0.85, 0.3 + features.peak_amplitude * 0.5), 3),
                "start_ms": 0,
                "end_ms": end_ms,
            }
        )

    return detections


def _pcm_to_mono_samples(
    raw_frames: bytes, num_channels: int, sample_width_bytes: int
) -> list[float]:
    bytes_per_frame = num_channels * sample_width_bytes
    if bytes_per_frame == 0:
        return []

    frame_total = len(raw_frames) // bytes_per_frame
    samples: list[float] = []

    for frame_index in range(frame_total):
        frame_offset = frame_index * bytes_per_frame
        channel_values: list[float] = []

        for channel_index in range(num_channels):
            start = frame_offset + (channel_index * sample_width_bytes)
            end = start + sample_width_bytes
            chunk = raw_frames[start:end]
            channel_values.append(_decode_sample(chunk, sample_width_bytes))

        samples.append(sum(channel_values) / len(channel_values))

    return samples


def _decode_sample(sample_bytes: bytes, sample_width_bytes: int) -> float:
    if sample_width_bytes == 1:
        raw_value = sample_bytes[0] - 128
        return raw_value / 128.0

    raw_value = int.from_bytes(sample_bytes, byteorder="little", signed=True)
    max_value = float(2 ** ((sample_width_bytes * 8) - 1))
    return raw_value / max_value
