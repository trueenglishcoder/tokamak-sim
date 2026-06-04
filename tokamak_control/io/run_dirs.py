"""Helpers for timestamped artifact run directories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time


@dataclass(frozen=True, slots=True)
class ArtifactRunDir:
    """Пути и числовой идентификатор директории с артефактами запуска."""

    run_dir: Path
    manifest_path: Path
    run_id: int


def slugify(value: str) -> str:
    """Преобразовать произвольную строку в безопасный фрагмент имени пути."""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "run"


def timestamped_stem(*parts: object) -> str:
    """Собрать имя директории из даты и смысловых частей запуска."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_parts = [slugify(str(part)) for part in parts if str(part).strip()]
    return "_".join([timestamp, *clean_parts])


def allocate_artifact_run_dir(root: str | Path, stem: str) -> ArtifactRunDir:
    """Создать уникальную директорию артефактов и manifest-путь внутри нее."""
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)

    base = slugify(stem)
    candidate = output_root / base
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix}"
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    run_id = int(time.time_ns())
    return ArtifactRunDir(
        run_dir=candidate,
        manifest_path=candidate / f"manifest{run_id}.json",
        run_id=run_id,
    )
