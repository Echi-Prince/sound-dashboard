import wave

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.audio import build_prototype_detections, decode_wav, extract_features
from app.config import settings
from app.schemas import (
    AnalysisResponse,
    AudioFeatures,
    AudioMetadata,
    DetectionResult,
    HealthResponse,
)


app = FastAPI(
    title="Sound Dashboard API",
    version="0.1.0",
    description=(
        "Backend scaffold for sound event detection and selective suppression."
    ),
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/config")
async def config() -> dict:
    return settings.model_dump()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="An uploaded file is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        decoded_audio = decode_wav(file_bytes)
    except (wave.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    features = extract_features(
        samples=decoded_audio.samples,
        sample_rate_hz=decoded_audio.sample_rate_hz,
    )
    detections = [
        DetectionResult(**detection)
        for detection in build_prototype_detections(
            features=features,
            duration_ms=decoded_audio.duration_ms,
        )
    ]

    return AnalysisResponse(
        filename=file.filename,
        status="prototype",
        message=(
            "Decoded WAV audio and computed prototype features. "
            "Current detections are heuristic and will be replaced by a model."
        ),
        metadata=AudioMetadata(
            sample_rate_hz=decoded_audio.sample_rate_hz,
            num_channels=decoded_audio.num_channels,
            sample_width_bytes=decoded_audio.sample_width_bytes,
            duration_ms=decoded_audio.duration_ms,
            frame_count=decoded_audio.frame_count,
        ),
        features=AudioFeatures(
            rms=features.rms,
            peak_amplitude=features.peak_amplitude,
            zero_crossing_rate=features.zero_crossing_rate,
            dominant_activity_ratio=features.dominant_activity_ratio,
        ),
        detections=detections,
    )
