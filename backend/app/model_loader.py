from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import ComputedFeatures, SpectralFeatures
from .classifier import (
    BaselineSoundClassifier,
    ClassPrediction,
    apply_prediction_thresholds,
)


@dataclass
class ModelArtifactManifest:
    model_name: str
    model_type: str
    class_names: list[str]
    sample_rate_hz: int
    input_sample_count: int
    confidence_threshold: float
    weights_path: str
    normalization_target_peak: float

    @property
    def weights_path_obj(self) -> Path:
        return Path(self.weights_path)


def load_model_manifest(manifest_path: str) -> ModelArtifactManifest | None:
    if not manifest_path.strip():
        return None

    path = Path(manifest_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelArtifactManifest(
        model_name=str(payload["model_name"]),
        model_type=str(payload["model_type"]),
        class_names=[str(label) for label in payload["class_names"]],
        sample_rate_hz=int(payload["sample_rate_hz"]),
        input_sample_count=int(payload["input_sample_count"]),
        confidence_threshold=float(payload["confidence_threshold"]),
        weights_path=str((path.parent / payload["weights_path"]).resolve()),
        normalization_target_peak=float(payload.get("normalization_target_peak", 0.95)),
    )


def prepare_waveform_input(
    samples: list[float], input_sample_count: int
) -> list[float]:
    if input_sample_count <= 0:
        return samples[:]

    prepared = samples[:input_sample_count]
    if len(prepared) < input_sample_count:
        prepared.extend([0.0] * (input_sample_count - len(prepared)))
    return prepared


class TorchscriptWaveformClassifier:
    def __init__(self, manifest: ModelArtifactManifest) -> None:
        self.manifest = manifest
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("torch is required to load a trained model artifact.") from exc

        self._torch = torch
        self._module = torch.jit.load(str(manifest.weights_path_obj), map_location="cpu")
        self._module.eval()

    def predict(self, samples: list[float]) -> list[ClassPrediction]:
        predictions = self.predict_ranked(samples)
        return [
            prediction
            for prediction in predictions
            if prediction.confidence >= self.manifest.confidence_threshold
        ][:3]

    def predict_ranked(self, samples: list[float]) -> list[ClassPrediction]:
        prepared_samples = prepare_waveform_input(
            samples=samples,
            input_sample_count=self.manifest.input_sample_count,
        )
        waveform = self._torch.tensor(
            prepared_samples,
            dtype=self._torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        with self._torch.no_grad():
            logits = self._module(waveform)
            probabilities = self._torch.softmax(logits, dim=1).squeeze(0).tolist()

        predictions = [
            ClassPrediction(label=label, confidence=round(float(confidence), 3))
            for label, confidence in zip(self.manifest.class_names, probabilities)
        ]
        predictions.sort(key=lambda item: item.confidence, reverse=True)
        return predictions


@dataclass
class PredictionResult:
    predictions: list[ClassPrediction]
    source_name: str
    used_fallback: bool


@dataclass
class TrainedPredictorEntry:
    name: str
    predictor: Any
    manifest: ModelArtifactManifest


@dataclass
class InferenceBackend:
    name: str
    predictor: Any
    fallback_predictor: Any | None = None
    manifest: ModelArtifactManifest | None = None
    fallback_reason: str = ""
    confidence_threshold: float = 0.45
    class_confidence_thresholds: dict[str, float] | None = None
    trained_predictors: list[TrainedPredictorEntry] | None = None

    def predict(
        self,
        samples: list[float],
        sample_rate_hz: int,
        features: ComputedFeatures,
        spectral_features: SpectralFeatures,
    ) -> list[ClassPrediction]:
        return self.predict_with_metadata(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            features=features,
            spectral_features=spectral_features,
        ).predictions

    def predict_with_metadata(
        self,
        samples: list[float],
        sample_rate_hz: int,
        features: ComputedFeatures,
        spectral_features: SpectralFeatures,
    ) -> PredictionResult:
        trained_candidates = self.trained_predictors or []
        if not trained_candidates and self.manifest is not None:
            trained_candidates = [
                TrainedPredictorEntry(
                    name=self.name,
                    predictor=self.predictor,
                    manifest=self.manifest,
                )
            ]

        for candidate_index, candidate in enumerate(trained_candidates):
            if (
                isinstance(candidate.predictor, TorchscriptWaveformClassifier)
                and sample_rate_hz == candidate.manifest.sample_rate_hz
            ):
                predictions = apply_prediction_thresholds(
                    predictions=candidate.predictor.predict_ranked(samples),
                    default_threshold=self.confidence_threshold,
                    class_confidence_thresholds=self.class_confidence_thresholds,
                )
                if predictions:
                    return PredictionResult(
                        predictions=predictions,
                        source_name=candidate.name,
                        used_fallback=candidate_index > 0,
                    )

        predictor = self.fallback_predictor or self.predictor
        if hasattr(predictor, "predict_ranked"):
            fallback_predictions = apply_prediction_thresholds(
                predictions=predictor.predict_ranked(
                    features=features,
                    spectral_features=spectral_features,
                ),
                default_threshold=self.confidence_threshold,
                class_confidence_thresholds=self.class_confidence_thresholds,
            )
        else:
            fallback_predictions = predictor.predict(
                features=features,
                spectral_features=spectral_features,
            )
        if not fallback_predictions and isinstance(predictor, BaselineSoundClassifier):
            fallback_predictions = predictor.predict_ranked(
                features=features,
                spectral_features=spectral_features,
            )[:1]
        fallback_name = self.name if predictor is self.predictor else "baseline_rules_v1"
        return PredictionResult(
            predictions=fallback_predictions,
            source_name=fallback_name,
            used_fallback=predictor is not self.predictor,
        )


def build_inference_backend(
    *,
    supported_classes: list[str],
    confidence_threshold: float,
    class_confidence_thresholds: dict[str, float] | None,
    baseline_name: str,
    manifest_paths: list[str] | None = None,
) -> InferenceBackend:
    baseline = BaselineSoundClassifier(
        supported_classes=supported_classes,
        confidence_threshold=confidence_threshold,
        class_confidence_thresholds=class_confidence_thresholds,
    )

    trained_predictors: list[TrainedPredictorEntry] = []
    for manifest_path in manifest_paths or []:
        manifest = load_model_manifest(manifest_path)
        if manifest is None:
            continue

        try:
            predictor = TorchscriptWaveformClassifier(manifest=manifest)
        except (OSError, RuntimeError, ValueError):
            continue

        trained_predictors.append(
            TrainedPredictorEntry(
                name=_build_predictor_name(manifest, manifest_path),
                predictor=predictor,
                manifest=manifest,
            )
        )

    if not trained_predictors:
        return InferenceBackend(
            name=baseline_name,
            predictor=baseline,
            confidence_threshold=confidence_threshold,
            class_confidence_thresholds=class_confidence_thresholds,
        )

    primary_predictor = trained_predictors[0]
    return InferenceBackend(
        name=primary_predictor.name,
        predictor=primary_predictor.predictor,
        fallback_predictor=baseline,
        manifest=primary_predictor.manifest,
        confidence_threshold=confidence_threshold,
        class_confidence_thresholds=class_confidence_thresholds,
        trained_predictors=trained_predictors,
    )


def _build_predictor_name(manifest: ModelArtifactManifest, manifest_path: str) -> str:
    manifest_parent = Path(manifest_path).resolve().parent.name
    return f"trained_model:{manifest.model_name}:{manifest_parent}"
