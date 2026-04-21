from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_inference_manifest_candidates() -> list[str]:
    active_manifest_path = get_active_manifest_path()
    candidate_paths: list[str] = []
    seen: set[str] = set()

    def add_candidate(path_value: str) -> None:
        if not path_value:
            return
        resolved = str(_resolve_path(path_value))
        if resolved in seen:
            return
        seen.add(resolved)
        candidate_paths.append(resolved)

    add_candidate(active_manifest_path)
    add_candidate(settings.trained_model_manifest_path)

    versions_dir = _resolve_path(settings.training_versions_dir)
    if versions_dir.exists():
        for manifest_path in sorted(
            versions_dir.glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            add_candidate(str(manifest_path))

    return candidate_paths


def list_model_artifacts() -> list[dict[str, object]]:
    active_state = get_active_model_state()
    active_manifest_path = str(_resolve_path(active_state["active_manifest_path"]))
    artifacts: list[dict[str, object]] = []
    seen: set[str] = set()

    for manifest_path in get_inference_manifest_candidates():
        resolved_manifest_path = str(_resolve_path(manifest_path))
        if resolved_manifest_path in seen:
            continue
        seen.add(resolved_manifest_path)
        manifest_file = Path(resolved_manifest_path)
        if not manifest_file.exists():
            continue
        artifacts.append(
            _build_artifact_summary(
                manifest_file,
                is_active=resolved_manifest_path == active_manifest_path,
                source_run_id=active_state["source_run_id"] if resolved_manifest_path == active_manifest_path else "",
            )
        )

    return artifacts


def activate_model_artifact(artifact_id: str) -> dict[str, object]:
    manifest_path = _resolve_artifact_id(artifact_id)
    set_active_manifest_path(str(manifest_path))
    return _build_artifact_summary(manifest_path, is_active=True, source_run_id="")


def get_active_manifest_path() -> str:
    state_path = _resolve_path(settings.active_model_state_path)
    if not state_path.exists():
        return settings.trained_model_manifest_path

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return settings.trained_model_manifest_path
    return str(payload.get("active_manifest_path") or settings.trained_model_manifest_path)


def set_active_manifest_path(manifest_path: str, *, source_run_id: str = "") -> None:
    state_path = _resolve_path(settings.active_model_state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "active_manifest_path": str(_resolve_path(manifest_path)),
                "source_run_id": source_run_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def get_active_model_state() -> dict[str, str]:
    state_path = _resolve_path(settings.active_model_state_path)
    if not state_path.exists():
        return {
            "active_manifest_path": str(_resolve_path(settings.trained_model_manifest_path)),
            "source_run_id": "",
        }
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "active_manifest_path": str(_resolve_path(settings.trained_model_manifest_path)),
            "source_run_id": "",
        }
    return {
        "active_manifest_path": str(payload.get("active_manifest_path") or _resolve_path(settings.trained_model_manifest_path)),
        "source_run_id": str(payload.get("source_run_id") or ""),
    }


def _build_artifact_summary(
    manifest_path: Path,
    *,
    is_active: bool,
    source_run_id: str,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = _relative_path(manifest_path)
    updated_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "artifact_id": _encode_artifact_id(relative_path),
        "model_name": str(payload.get("model_name", manifest_path.parent.name)),
        "relative_path": relative_path,
        "weights_relative_path": _relative_path((manifest_path.parent / payload.get("weights_path", "model.ts")).resolve()),
        "class_names": [str(label) for label in payload.get("class_names", [])],
        "training_example_count": int(payload.get("training_example_count", 0)),
        "validation_example_count": int(payload.get("validation_example_count", 0)),
        "updated_at": updated_at,
        "is_active": is_active,
        "source_run_id": source_run_id,
    }


def _resolve_artifact_id(artifact_id: str) -> Path:
    try:
        padded_value = artifact_id + ("=" * ((4 - len(artifact_id) % 4) % 4))
        relative_path = base64.urlsafe_b64decode(padded_value.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise FileNotFoundError(artifact_id) from exc

    manifest_path = (_REPO_ROOT / relative_path).resolve()
    if not manifest_path.exists() or manifest_path.suffix.lower() != ".json":
        raise FileNotFoundError(artifact_id)
    try:
        manifest_path.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise FileNotFoundError(artifact_id) from exc
    return manifest_path


def _encode_artifact_id(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(_REPO_ROOT).as_posix()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()
