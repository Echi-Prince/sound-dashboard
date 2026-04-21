from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.app.audio import decode_wav
from training.dataset import build_examples_from_labeled_directory, write_manifest

from .config import settings
from .recording_store import list_training_recordings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_dataset_summary() -> dict[str, object]:
    recordings = list_training_recordings()
    by_label = {label: 0 for label in settings.supported_classes}
    by_split = {"unspecified": 0, "train": 0, "val": 0, "test": 0}
    per_label_splits = {
        label: {"unspecified": 0, "train": 0, "val": 0, "test": 0}
        for label in settings.supported_classes
    }

    total_duration_ms = 0
    for recording in recordings:
        label = str(recording["label"])
        split = str(recording["split"] or "unspecified")
        by_label[label] = by_label.get(label, 0) + 1
        by_split[split] = by_split.get(split, 0) + 1
        label_splits = per_label_splits.setdefault(
            label,
            {"unspecified": 0, "train": 0, "val": 0, "test": 0},
        )
        label_splits[split] = label_splits.get(split, 0) + 1
        total_duration_ms += int(recording["duration_ms"])

    manifest_path = _manifest_path()
    manifest_exists = manifest_path.exists()
    return {
        "total_recordings": len(recordings),
        "total_duration_ms": total_duration_ms,
        "by_label": by_label,
        "by_split": by_split,
        "per_label_splits": per_label_splits,
        "manifest_relative_path": _relative_path(manifest_path),
        "manifest_exists": manifest_exists,
        "manifest_updated_at": _isoformat(manifest_path.stat().st_mtime) if manifest_exists else "",
    }


def build_real_recordings_manifest() -> dict[str, object]:
    source_dir = _recordings_dir()
    manifest_path = _manifest_path()
    examples = build_examples_from_labeled_directory(
        str(source_dir),
        supported_classes=settings.supported_classes,
        validation_ratio=settings.real_recordings_validation_ratio,
        test_ratio=settings.real_recordings_test_ratio,
        seed=settings.real_recordings_split_seed,
    )

    readable_examples = []
    skipped_examples = []
    for example in examples:
        try:
            decode_wav(example.audio_path.read_bytes())
            readable_examples.append(example)
        except Exception as exc:
            skipped_examples.append((example, str(exc)))

    if not readable_examples:
        raise ValueError(
            "No readable WAV files were found. Add recordings to training/real_recordings/ and retry."
        )

    write_manifest(
        examples=readable_examples,
        manifest_path=str(manifest_path),
        base_dir=str(source_dir),
    )

    by_split_counter = Counter(example.split for example in readable_examples)
    by_label_counter = Counter(example.label for example in readable_examples)
    per_label_splits_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for example in readable_examples:
        per_label_splits_counter[example.label][example.split] += 1

    return {
        "manifest_relative_path": _relative_path(manifest_path),
        "total_examples": len(readable_examples),
        "by_split": {
            split: by_split_counter.get(split, 0)
            for split in ("train", "val", "test")
            if by_split_counter.get(split, 0) > 0
        },
        "by_label": {
            label: by_label_counter.get(label, 0)
            for label in settings.supported_classes
            if by_label_counter.get(label, 0) > 0
        },
        "per_label_splits": {
            label: {
                split: per_label_splits_counter[label].get(split, 0)
                for split in ("train", "val", "test")
            }
            for label in settings.supported_classes
            if per_label_splits_counter.get(label)
        },
        "skipped_count": len(skipped_examples),
        "skipped_files": [
            {
                "relative_path": _relative_path(example.audio_path),
                "reason": reason,
            }
            for example, reason in skipped_examples
        ],
        "updated_at": _isoformat(manifest_path.stat().st_mtime),
    }


def _recordings_dir() -> Path:
    configured_path = Path(settings.real_recordings_dir)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _manifest_path() -> Path:
    configured_path = Path(settings.real_recordings_manifest_path)
    if configured_path.is_absolute():
        return configured_path
    return (_REPO_ROOT / configured_path).resolve()


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(_REPO_ROOT).as_posix()


def _isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
