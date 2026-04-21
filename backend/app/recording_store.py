from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .audio import decode_wav
from .config import settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_SPLITS = {"train", "val", "test"}


def save_training_recording(
    *,
    file_bytes: bytes,
    label: str,
    split: str = "",
    source_name: str = "",
) -> dict[str, int | str]:
    normalized_label = _validate_label(label)
    normalized_split = _validate_split(split)

    safe_source = _slugify(source_name) or "browser"
    recording_uuid = uuid4().hex
    filename = f"{normalized_label}-{safe_source}-{recording_uuid[:8]}.wav"

    target_dir = _recordings_dir()
    if normalized_split:
        target_dir = target_dir / normalized_split
    target_dir = target_dir / normalized_label
    target_dir.mkdir(parents=True, exist_ok=True)

    absolute_path = target_dir / filename
    absolute_path.write_bytes(file_bytes)
    return build_recording_summary(absolute_path)


def list_training_recordings() -> list[dict[str, int | str | bool]]:
    recordings_dir = _recordings_dir()
    if not recordings_dir.exists():
        return []

    summaries: list[dict[str, int | str | bool]] = []
    for wav_path in recordings_dir.rglob("*.wav"):
        if wav_path.is_file():
            try:
                summaries.append(build_recording_summary(wav_path))
            except (ValueError, OSError):
                continue

    summaries.sort(
        key=lambda item: (str(item["updated_at"]), str(item["filename"])),
        reverse=True,
    )
    return summaries


def load_training_recording(recording_id: str) -> dict[str, int | str | bool]:
    recording_path = _resolve_recording_id(recording_id)
    return build_recording_detail(recording_path)


def update_training_recording(
    *,
    recording_id: str,
    label: str,
    split: str = "",
) -> dict[str, int | str | bool]:
    recording_path = _resolve_recording_id(recording_id)
    normalized_label = _validate_label(label)
    normalized_split = _validate_split(split)

    target_dir = _recordings_dir()
    if normalized_split:
        target_dir = target_dir / normalized_split
    target_dir = target_dir / normalized_label
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / recording_path.name
    if target_path.resolve() != recording_path.resolve():
        recording_path.replace(target_path)
        _cleanup_empty_parent_dirs(recording_path.parent)
    return build_recording_summary(target_path)


def delete_training_recording(recording_id: str) -> None:
    recording_path = _resolve_recording_id(recording_id)
    parent_dir = recording_path.parent
    recording_path.unlink(missing_ok=False)
    _cleanup_empty_parent_dirs(parent_dir)


def build_recording_summary(recording_path: Path) -> dict[str, int | str | bool]:
    decoded_audio = decode_wav(recording_path.read_bytes())
    relative_path = recording_path.relative_to(_REPO_ROOT).as_posix()
    stat_result = recording_path.stat()
    split, label = _extract_split_and_label(recording_path)
    return {
        "recording_id": _encode_recording_id(relative_path),
        "label": label,
        "split": split,
        "filename": recording_path.name,
        "relative_path": relative_path,
        "byte_count": stat_result.st_size,
        "duration_ms": decoded_audio.duration_ms,
        "sample_rate_hz": decoded_audio.sample_rate_hz,
        "created_at": _to_isoformat(stat_result.st_ctime),
        "updated_at": _to_isoformat(stat_result.st_mtime),
        "source_name": _extract_source_name(recording_path.stem, label),
    }


def build_recording_detail(recording_path: Path) -> dict[str, int | str | bool]:
    summary = build_recording_summary(recording_path)
    file_bytes = recording_path.read_bytes()
    summary["wav_base64"] = base64.b64encode(file_bytes).decode("ascii")
    return summary


def _resolve_recording_id(recording_id: str) -> Path:
    try:
        padded_value = recording_id + ("=" * ((4 - len(recording_id) % 4) % 4))
        relative_path = base64.urlsafe_b64decode(padded_value.encode("ascii")).decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise FileNotFoundError(recording_id) from exc

    target_path = (_REPO_ROOT / relative_path).resolve()
    recordings_root = _recordings_dir().resolve()
    try:
        target_path.relative_to(recordings_root)
    except ValueError as exc:
        raise FileNotFoundError(recording_id) from exc
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(recording_id)
    return target_path


def _recordings_dir() -> Path:
    configured_path = Path(settings.real_recordings_dir)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _validate_label(label: str) -> str:
    normalized_label = label.strip().lower()
    if normalized_label not in settings.supported_classes:
        raise ValueError("label must be one of the supported classes.")
    return normalized_label


def _validate_split(split: str) -> str:
    normalized_split = split.strip().lower()
    if normalized_split and normalized_split not in _VALID_SPLITS:
        raise ValueError("split must be blank or one of: train, val, test.")
    return normalized_split


def _extract_split_and_label(recording_path: Path) -> tuple[str, str]:
    relative_parts = recording_path.relative_to(_recordings_dir()).parts
    if len(relative_parts) >= 3 and relative_parts[0] in _VALID_SPLITS:
        return relative_parts[0], relative_parts[1]
    if len(relative_parts) >= 2:
        return "", relative_parts[0]
    raise ValueError("Recording path does not match the expected layout.")


def _extract_source_name(stem: str, label: str) -> str:
    if not stem.startswith(f"{label}-"):
        return ""
    trimmed = stem[len(label) + 1 :]
    pieces = trimmed.split("-")
    if len(pieces) > 1 and len(pieces[-1]) == 8:
        pieces = pieces[:-1]
    return "-".join(piece for piece in pieces if piece)


def _cleanup_empty_parent_dirs(path: Path) -> None:
    recordings_root = _recordings_dir().resolve()
    current = path.resolve()
    while current != recordings_root and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _encode_recording_id(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")


def _to_isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
