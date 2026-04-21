from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from training.train import train_waveform_model

from .artifact_manager import set_active_manifest_path
from .config import settings
from .inference_manager import clear_inference_backend_cache

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE_LOCK = threading.Lock()
_TRAINING_THREAD: threading.Thread | None = None
_LATEST_STATE: dict[str, object] | None = None


def get_training_status() -> dict[str, object]:
    with _STATE_LOCK:
        if _LATEST_STATE is not None:
            return dict(_LATEST_STATE)
    persisted_state = _load_state_file()
    return persisted_state or _default_state()


def start_training_run() -> dict[str, object]:
    manifest_path = _resolve_path(settings.real_recordings_manifest_path)
    if not manifest_path.exists():
        raise ValueError("Build the real-recordings manifest before starting training.")

    global _TRAINING_THREAD
    with _STATE_LOCK:
        current_state = _LATEST_STATE or _load_state_file() or _default_state()
        if _TRAINING_THREAD is not None and _TRAINING_THREAD.is_alive():
            raise RuntimeError("A training run is already in progress.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    next_state = {
        "status": "running",
        "run_id": run_id,
        "started_at": _utc_now(),
        "finished_at": "",
        "manifest_relative_path": _relative_path(manifest_path),
        "output_relative_path": _relative_path(_resolve_path(settings.training_output_dir)),
        "epochs": settings.training_epochs,
        "batch_size": settings.training_batch_size,
        "learning_rate": settings.training_learning_rate,
        "current_epoch": 0,
        "last_loss": 0.0,
        "last_val_accuracy": 0.0,
        "final_val_accuracy": current_state.get("final_val_accuracy", 0.0),
        "message": "Training started.",
        "error": "",
    }
    _write_state(next_state)

    _TRAINING_THREAD = threading.Thread(
        target=_run_training_job,
        kwargs={"run_id": run_id},
        daemon=True,
        name="sound-dashboard-training",
    )
    _TRAINING_THREAD.start()
    return dict(next_state)


def _run_training_job(*, run_id: str) -> None:
    manifest_path = _resolve_path(settings.real_recordings_manifest_path)
    output_dir = _resolve_path(settings.training_output_dir)
    try:
        result = train_waveform_model(
            manifest_path=str(manifest_path),
            output_dir=str(output_dir),
            epochs=settings.training_epochs,
            batch_size=settings.training_batch_size,
            learning_rate=settings.training_learning_rate,
            progress_callback=lambda epoch_result: _update_progress(run_id, epoch_result),
        )
        version_manifest_path = _archive_training_output(run_id=run_id, output_dir=output_dir)
        set_active_manifest_path(str(version_manifest_path), source_run_id=run_id)
        clear_inference_backend_cache()
        _write_state(
            {
                "status": "completed",
                "run_id": run_id,
                "started_at": get_training_status().get("started_at", ""),
                "finished_at": _utc_now(),
                "manifest_relative_path": _relative_path(manifest_path),
                "output_relative_path": _relative_path(version_manifest_path.parent),
                "epochs": settings.training_epochs,
                "batch_size": settings.training_batch_size,
                "learning_rate": settings.training_learning_rate,
                "current_epoch": len(result["epoch_metrics"]),
                "last_loss": float(result["epoch_metrics"][-1]["loss"]) if result["epoch_metrics"] else 0.0,
                "last_val_accuracy": float(result["final_val_accuracy"]),
                "final_val_accuracy": float(result["final_val_accuracy"]),
                "message": "Training completed successfully.",
                "error": "",
            }
        )
    except Exception as exc:
        previous_state = get_training_status()
        _write_state(
            {
                "status": "failed",
                "run_id": run_id,
                "started_at": previous_state.get("started_at", ""),
                "finished_at": _utc_now(),
                "manifest_relative_path": _relative_path(manifest_path) if manifest_path.exists() else "",
                "output_relative_path": _relative_path(output_dir),
                "epochs": settings.training_epochs,
                "batch_size": settings.training_batch_size,
                "learning_rate": settings.training_learning_rate,
                "current_epoch": int(previous_state.get("current_epoch", 0)),
                "last_loss": float(previous_state.get("last_loss", 0.0)),
                "last_val_accuracy": float(previous_state.get("last_val_accuracy", 0.0)),
                "final_val_accuracy": float(previous_state.get("final_val_accuracy", 0.0)),
                "message": "Training failed.",
                "error": str(exc),
            }
        )


def _update_progress(run_id: str, epoch_result: dict[str, float | int]) -> None:
    current_state = get_training_status()
    _write_state(
        {
            "status": "running",
            "run_id": run_id,
            "started_at": current_state.get("started_at", ""),
            "finished_at": "",
            "manifest_relative_path": current_state.get("manifest_relative_path", ""),
            "output_relative_path": current_state.get("output_relative_path", ""),
            "epochs": settings.training_epochs,
            "batch_size": settings.training_batch_size,
            "learning_rate": settings.training_learning_rate,
            "current_epoch": int(epoch_result["epoch"]),
            "last_loss": float(epoch_result["loss"]),
            "last_val_accuracy": float(epoch_result["val_accuracy"]),
            "final_val_accuracy": float(current_state.get("final_val_accuracy", 0.0)),
            "message": f"Training epoch {int(epoch_result['epoch'])} of {settings.training_epochs}.",
            "error": "",
        }
    )


def _default_state() -> dict[str, object]:
    return {
        "status": "idle",
        "run_id": "",
        "started_at": "",
        "finished_at": "",
        "manifest_relative_path": "",
        "output_relative_path": _relative_path(_resolve_path(settings.training_output_dir)),
        "epochs": settings.training_epochs,
        "batch_size": settings.training_batch_size,
        "learning_rate": settings.training_learning_rate,
        "current_epoch": 0,
        "last_loss": 0.0,
        "last_val_accuracy": 0.0,
        "final_val_accuracy": 0.0,
        "message": "Training has not started yet.",
        "error": "",
    }


def _state_file_path() -> Path:
    runs_dir = _resolve_path(settings.training_runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / "latest-status.json"


def _write_state(state: dict[str, object]) -> None:
    with _STATE_LOCK:
        global _LATEST_STATE
        _LATEST_STATE = dict(state)
        _state_file_path().write_text(json.dumps(_LATEST_STATE, indent=2), encoding="utf-8")


def _load_state_file() -> dict[str, object] | None:
    state_path = _state_file_path()
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _archive_training_output(*, run_id: str, output_dir: Path) -> Path:
    versions_dir = _resolve_path(settings.training_versions_dir)
    versions_dir.mkdir(parents=True, exist_ok=True)
    target_dir = versions_dir / run_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(output_dir, target_dir)
    return target_dir / "manifest.json"


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(_REPO_ROOT).as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
