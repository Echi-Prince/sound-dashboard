from pydantic import BaseModel


class AppConfig(BaseModel):
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 1000
    chunk_overlap_ms: int = 500
    classifier_name: str = "baseline_rules_v1"
    classifier_confidence_threshold: float = 0.45
    class_confidence_thresholds: dict[str, float] = {
        "speech": 0.55,
        "keyboard": 0.4,
        "dog_bark": 0.55,
        "traffic": 0.45,
        "siren": 0.5,
        "vacuum": 0.45,
        "music": 0.5,
    }
    fallback_dominant_label_share_threshold: float = 0.5
    fallback_secondary_label_share_threshold: float = 0.3
    default_suppression_factor: float = 0.2
    trained_model_manifest_path: str = "training/artifacts/real-v1/manifest.json"
    real_recordings_dir: str = "training/real_recordings"
    real_recordings_manifest_path: str = "training/real_recordings/manifest.jsonl"
    real_recordings_validation_ratio: float = 0.2
    real_recordings_test_ratio: float = 0.1
    real_recordings_split_seed: int = 7
    training_runs_dir: str = "training/artifacts/runs"
    training_output_dir: str = "training/artifacts/latest-auto"
    training_versions_dir: str = "training/artifacts/versions"
    active_model_state_path: str = "training/artifacts/active-model.json"
    training_epochs: int = 8
    training_batch_size: int = 8
    training_learning_rate: float = 0.001
    session_store_dir: str = "backend/data/sessions"
    session_list_limit: int = 20
    supported_classes: list[str] = [
        "speech",
        "keyboard",
        "dog_bark",
        "traffic",
        "siren",
        "vacuum",
        "music",
    ]


settings = AppConfig()
