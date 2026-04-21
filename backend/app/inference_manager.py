from __future__ import annotations

from .artifact_manager import get_inference_manifest_candidates
from .config import settings
from .model_loader import InferenceBackend, build_inference_backend

_CACHED_KEY: tuple[str, ...] | None = None
_CACHED_BACKEND: InferenceBackend | None = None


def get_inference_backend() -> InferenceBackend:
    global _CACHED_BACKEND, _CACHED_KEY
    manifest_candidates = tuple(get_inference_manifest_candidates())
    if _CACHED_BACKEND is not None and _CACHED_KEY == manifest_candidates:
        return _CACHED_BACKEND

    _CACHED_BACKEND = build_inference_backend(
        supported_classes=settings.supported_classes,
        confidence_threshold=settings.classifier_confidence_threshold,
        class_confidence_thresholds=settings.class_confidence_thresholds,
        baseline_name=settings.classifier_name,
        manifest_paths=list(manifest_candidates),
    )
    _CACHED_KEY = manifest_candidates
    return _CACHED_BACKEND


def clear_inference_backend_cache() -> None:
    global _CACHED_BACKEND, _CACHED_KEY
    _CACHED_BACKEND = None
    _CACHED_KEY = None
