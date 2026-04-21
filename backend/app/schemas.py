from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AudioMetadata(BaseModel):
    sample_rate_hz: int
    processed_sample_rate_hz: int
    num_channels: int
    sample_width_bytes: int
    duration_ms: int
    frame_count: int
    processed_sample_count: int
    was_resampled: bool
    original_peak_amplitude: float
    normalized_peak_amplitude: float
    normalization_gain: float


class AudioFeatures(BaseModel):
    rms: float
    peak_amplitude: float
    zero_crossing_rate: float
    dominant_activity_ratio: float


class SpectralFeatures(BaseModel):
    frame_count: int
    mel_bin_count: int
    frame_size: int
    hop_size: int
    fft_size: int
    min_db: float
    max_db: float
    mean_db: float
    dynamic_range_db: float
    low_band_mean_db: float
    mid_band_mean_db: float
    high_band_mean_db: float


class DetectionResult(BaseModel):
    label: str
    confidence: float
    start_ms: int
    end_ms: int


class ProcessedAudio(BaseModel):
    sample_rate_hz: int
    duration_ms: int
    sample_width_bytes: int
    wav_byte_count: int
    attenuation_factor: float
    suppressed_classes: list[str]
    class_attenuation_factors: dict[str, float]
    wav_base64: str


class AnalysisResponse(BaseModel):
    session_id: str
    filename: str
    status: str
    classifier_source: str
    used_fallback: bool
    message: str
    metadata: AudioMetadata
    features: AudioFeatures
    spectral_features: SpectralFeatures
    detections: list[DetectionResult]


class ProcessResponse(BaseModel):
    session_id: str
    filename: str
    status: str
    classifier_source: str
    used_fallback: bool
    message: str
    metadata: AudioMetadata
    detections: list[DetectionResult]
    processed_audio: ProcessedAudio


class SavedRecordingResponse(BaseModel):
    recording_id: str
    label: str
    split: str
    filename: str
    relative_path: str
    byte_count: int
    duration_ms: int
    sample_rate_hz: int
    message: str


class RecordingListItem(BaseModel):
    recording_id: str
    label: str
    split: str
    filename: str
    relative_path: str
    byte_count: int
    duration_ms: int
    sample_rate_hz: int
    created_at: str
    updated_at: str
    source_name: str


class RecordingListResponse(BaseModel):
    recordings: list[RecordingListItem]


class RecordingDetailResponse(RecordingListItem):
    wav_base64: str


class RecordingUpdateRequest(BaseModel):
    label: str
    split: str = ""


class DatasetSummaryResponse(BaseModel):
    total_recordings: int
    total_duration_ms: int
    by_label: dict[str, int]
    by_split: dict[str, int]
    per_label_splits: dict[str, dict[str, int]]
    manifest_relative_path: str
    manifest_exists: bool
    manifest_updated_at: str


class ManifestSkippedFile(BaseModel):
    relative_path: str
    reason: str


class BuildManifestResponse(BaseModel):
    manifest_relative_path: str
    total_examples: int
    by_split: dict[str, int]
    by_label: dict[str, int]
    per_label_splits: dict[str, dict[str, int]]
    skipped_count: int
    skipped_files: list[ManifestSkippedFile]
    updated_at: str
    message: str


class TrainingStatusResponse(BaseModel):
    status: str
    run_id: str
    started_at: str
    finished_at: str
    manifest_relative_path: str
    output_relative_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    current_epoch: int
    last_loss: float
    last_val_accuracy: float
    final_val_accuracy: float
    message: str
    error: str


class ArtifactListItem(BaseModel):
    artifact_id: str
    model_name: str
    relative_path: str
    weights_relative_path: str
    class_names: list[str]
    training_example_count: int
    validation_example_count: int
    updated_at: str
    is_active: bool
    source_run_id: str


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactListItem]


class ActivateArtifactRequest(BaseModel):
    artifact_id: str


class ActivateArtifactResponse(ArtifactListItem):
    message: str


class SessionListItem(BaseModel):
    session_id: str
    filename: str
    created_at: str
    updated_at: str
    status: str
    detection_count: int
    has_processed_audio: bool


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class SessionDetailResponse(BaseModel):
    session_id: str
    filename: str
    created_at: str
    updated_at: str
    original_audio_base64: str
    analysis: AnalysisResponse
    processed_response: ProcessResponse | None
