from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AudioMetadata(BaseModel):
    sample_rate_hz: int
    num_channels: int
    sample_width_bytes: int
    duration_ms: int
    frame_count: int


class AudioFeatures(BaseModel):
    rms: float
    peak_amplitude: float
    zero_crossing_rate: float
    dominant_activity_ratio: float


class DetectionResult(BaseModel):
    label: str
    confidence: float
    start_ms: int
    end_ms: int


class AnalysisResponse(BaseModel):
    filename: str
    status: str
    message: str
    metadata: AudioMetadata
    features: AudioFeatures
    detections: list[DetectionResult]
