"""Синтетические reference-траектории тока плазмы Ip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True, repr=True)
class IpTemplate:
    """Исходная таблица Ip, используемая как шаблон формы сигнала."""

    shot_id: str
    path: Path | None
    time_s: np.ndarray
    ip_a: np.ndarray

    def __post_init__(self) -> None:
        """Проверить физический формат шаблонной таблицы."""
        time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        ip_a = np.asarray(self.ip_a, dtype=float).reshape(-1)
        if time_s.shape != ip_a.shape:
            raise ValueError(f"Ip template time and Ip shapes differ: {time_s.shape} vs {ip_a.shape}")
        if time_s.size < 2:
            raise ValueError("Ip template must contain at least 2 rows")
        if not np.all(np.isfinite(time_s)):
            raise ValueError("Ip template contains non-finite timestamps")
        if not np.all(np.isfinite(ip_a)):
            raise ValueError("Ip template contains non-finite Ip values")
        if np.any(np.diff(time_s) < 0.0):
            raise ValueError("Ip template timestamps must be nondecreasing")
        if float(time_s[-1] - time_s[0]) <= 0.0:
            raise ValueError("Ip template duration must be positive")
        if not np.any(np.abs(ip_a) > 0.0):
            raise ValueError("Ip template must contain at least one nonzero Ip sample")
        object.__setattr__(self, "time_s", time_s - float(time_s[0]))
        object.__setattr__(self, "ip_a", ip_a)
        object.__setattr__(self, "path", None if self.path is None else Path(self.path))
        object.__setattr__(self, "shot_id", str(self.shot_id))

    @property
    def rows(self) -> int:
        """Вернуть число строк в таблице."""
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Вернуть длительность таблицы в секундах."""
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def peak_abs_ip(self) -> float:
        """Вернуть максимальный модуль тока плазмы."""
        return float(np.max(np.abs(self.ip_a)))


@dataclass(frozen=True, slots=True, repr=True)
class SyntheticIpConfig:
    """Настройки генерации Ip из реального шаблона."""

    amplitude_jitter: float = 0.05
    duration_jitter: float = 0.05
    shape_jitter: float = 0.02
    shape_anchors: int = 5
    start_value: float | None = None

    def __post_init__(self) -> None:
        """Проверить диапазоны jitter-параметров."""
        for name in ("amplitude_jitter", "duration_jitter", "shape_jitter"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        anchors = int(self.shape_anchors)
        if anchors < 2:
            raise ValueError("shape_anchors must be >= 2")
        if self.start_value is not None:
            start = float(self.start_value)
            if not np.isfinite(start):
                raise ValueError("start_value must be finite when provided")
            object.__setattr__(self, "start_value", start)
        object.__setattr__(self, "shape_anchors", anchors)


@dataclass(frozen=True, slots=True, repr=True)
class IpSegment:
    """One continuous segment in a generated Ip reference trajectory."""

    kind: str
    start_step: int
    end_step: int
    start_value: float
    end_value: float

    def __post_init__(self) -> None:
        if self.kind not in {"hold", "ramp"}:
            raise ValueError("IpSegment.kind must be 'hold' or 'ramp'")
        if int(self.start_step) < 0 or int(self.end_step) <= int(self.start_step):
            raise ValueError("IpSegment step range must satisfy end_step > start_step >= 0")
        if not np.isfinite(float(self.start_value)) or not np.isfinite(float(self.end_value)):
            raise ValueError("IpSegment values must be finite")


@dataclass(frozen=True, slots=True, repr=True)
class SegmentedIpConfig:
    """Settings for continuous segment-based synthetic Ip trajectories."""

    value_bounds: tuple[float, float]
    segment_step_bounds: tuple[int, int]
    segment_count_bounds: tuple[int, int]
    max_steps: int
    t_step: float
    rate_limit: float | None = None
    hold_probability: float = 0.35
    start_value: float | None = None

    def __post_init__(self) -> None:
        lo, hi = (float(self.value_bounds[0]), float(self.value_bounds[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("value_bounds must satisfy finite hi > lo")
        step_lo, step_hi = (int(self.segment_step_bounds[0]), int(self.segment_step_bounds[1]))
        if step_lo <= 0 or step_hi < step_lo:
            raise ValueError("segment_step_bounds must satisfy hi >= lo > 0")
        count_lo, count_hi = (int(self.segment_count_bounds[0]), int(self.segment_count_bounds[1]))
        if count_lo <= 0 or count_hi < count_lo:
            raise ValueError("segment_count_bounds must satisfy hi >= lo > 0")
        max_steps = int(self.max_steps)
        if max_steps < 2:
            raise ValueError("max_steps must be >= 2")
        t_step = float(self.t_step)
        if not np.isfinite(t_step) or t_step <= 0.0:
            raise ValueError("t_step must be finite and > 0")
        if self.rate_limit is not None and (not np.isfinite(float(self.rate_limit)) or float(self.rate_limit) <= 0.0):
            raise ValueError("rate_limit must be finite and > 0 when provided")
        hold_probability = float(self.hold_probability)
        if not np.isfinite(hold_probability) or hold_probability < 0.0 or hold_probability > 1.0:
            raise ValueError("hold_probability must be in [0, 1]")
        if self.start_value is not None:
            start = float(self.start_value)
            if not np.isfinite(start):
                raise ValueError("start_value must be finite when provided")
            object.__setattr__(self, "start_value", start)
        object.__setattr__(self, "value_bounds", (lo, hi))
        object.__setattr__(self, "segment_step_bounds", (step_lo, step_hi))
        object.__setattr__(self, "segment_count_bounds", (count_lo, count_hi))
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "t_step", t_step)
        object.__setattr__(self, "hold_probability", hold_probability)


@dataclass(frozen=True, slots=True, repr=True)
class IpReferenceTrajectory:
    """Готовая reference-траектория Ip с clamped-интерполяцией."""

    time_s: np.ndarray
    ip_a: np.ndarray
    template_shot_id: str | None = None
    amplitude_scale: float = 1.0
    duration_scale: float = 1.0
    segments: tuple[IpSegment, ...] = ()

    def __post_init__(self) -> None:
        """Проверить таблицу перед использованием в сценарии или RL bridge."""
        time_s = np.asarray(self.time_s, dtype=float).reshape(-1)
        ip_a = np.asarray(self.ip_a, dtype=float).reshape(-1)
        if time_s.shape != ip_a.shape:
            raise ValueError(f"Ip trajectory time and Ip shapes differ: {time_s.shape} vs {ip_a.shape}")
        if time_s.size < 2:
            raise ValueError("Ip trajectory must contain at least 2 rows")
        if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(ip_a)):
            raise ValueError("Ip trajectory must contain only finite values")
        if np.any(np.diff(time_s) < 0.0):
            raise ValueError("Ip trajectory timestamps must be nondecreasing")
        if float(time_s[-1] - time_s[0]) <= 0.0:
            raise ValueError("Ip trajectory duration must be positive")
        for name in ("amplitude_scale", "duration_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "ip_a", ip_a)
        if self.template_shot_id is not None:
            object.__setattr__(self, "template_shot_id", str(self.template_shot_id))
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def rows(self) -> int:
        """Вернуть число samples."""
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        """Вернуть длительность reference в секундах."""
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def peak_abs_ip(self) -> float:
        """Вернуть пиковый модуль Ip."""
        return float(np.max(np.abs(self.ip_a)))

    def at_time(self, t: float) -> float:
        """Вернуть Ip(t) с удержанием крайних значений вне таблицы."""
        return float(np.interp(float(t), self.time_s, self.ip_a, left=self.ip_a[0], right=self.ip_a[-1]))

    def table(self) -> np.ndarray:
        """Вернуть двухколоночную таблицу time;Ip."""
        return np.column_stack([self.time_s, self.ip_a])


def extract_t15_shot_id(path: str | Path) -> str:
    """Вытащить номер shot из имени файла формата t15md_<shot>_ip.csv."""
    p = Path(path)
    stem = p.stem.lower()
    prefix = "t15md_"
    suffix = "_ip"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unsupported Ip filename: {p.name}")
    shot_id = stem[len(prefix) : -len(suffix)]
    if not shot_id.isdigit():
        raise ValueError(f"Could not parse shot id from filename: {p.name}")
    return shot_id


def _read_semicolon_ip_array(path: str | Path) -> np.ndarray:
    """Прочитать raw time;Ip таблицу без нормализации времени."""
    p = Path(path)
    rows: list[list[float]] = []
    with p.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part for part in line.split(";") if part != ""]
            if len(parts) != 2:
                raise ValueError(f"Expected 2 semicolon-separated columns in {p}, got {len(parts)}")
            rows.append([float(parts[0]), float(parts[1])])
    if len(rows) < 2:
        raise ValueError(f"Ip table must contain at least 2 rows: {p}")
    return np.asarray(rows, dtype=float)


def load_semicolon_ip_table(path: str | Path, *, shot_id: str | None = None) -> IpTemplate:
    """Прочитать двухколоночную таблицу time;Ip в каноническом формате."""
    p = Path(path)
    arr = _read_semicolon_ip_array(p)
    return IpTemplate(
        shot_id=extract_t15_shot_id(p) if shot_id is None else str(shot_id),
        path=p,
        time_s=np.asarray(arr[:, 0], dtype=float),
        ip_a=np.asarray(arr[:, 1], dtype=float),
    )


def discover_ip_templates(source_ip_dir: str | Path) -> list[IpTemplate]:
    """Собрать и проверить все шаблонные Ip-таблицы из директории."""
    source = Path(source_ip_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source Ip directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source Ip path is not a directory: {source}")
    templates = [load_semicolon_ip_table(path) for path in sorted(source.glob("t15md_*_ip.csv"))]
    if not templates:
        raise FileNotFoundError(f"No t15md_*_ip.csv files found in {source}")
    return templates


def make_ip_reference_from_table(path: str | Path, *, time_offset: float | None = None) -> IpReferenceTrajectory:
    """Построить reference Ip из готовой таблицы без случайного изменения формы."""
    p = Path(path)
    if time_offset is None:
        template = load_semicolon_ip_table(p)
        return IpReferenceTrajectory(template.time_s, template.ip_a, template_shot_id=template.shot_id)
    origin = float(time_offset)
    if not np.isfinite(origin):
        raise ValueError("time_offset must be finite when provided")
    arr = _read_semicolon_ip_array(p)
    return IpReferenceTrajectory(
        np.asarray(arr[:, 0], dtype=float) - origin,
        np.asarray(arr[:, 1], dtype=float),
        template_shot_id=extract_t15_shot_id(p),
    )


def generate_ip_reference_from_template(
    template: IpTemplate,
    *,
    rng: np.random.Generator,
    config: SyntheticIpConfig | None = None,
) -> IpReferenceTrajectory:
    """Построить новую Ip-траекторию, сохранив базовую форму и единицы шаблона."""
    cfg = SyntheticIpConfig() if config is None else config
    duration_scale = _sample_scale(rng, jitter=cfg.duration_jitter)
    amplitude_scale = _sample_scale(rng, jitter=cfg.amplitude_jitter)

    rows = template.rows
    source_duration = template.duration_s
    target_duration = float(source_duration * duration_scale)
    query_phase = np.linspace(0.0, 1.0, rows, dtype=float)
    source_phase = template.time_s / float(source_duration)
    ip_base = np.interp(query_phase, source_phase, template.ip_a)
    envelope = _shape_envelope(rows, rng, shape_jitter=cfg.shape_jitter, n_anchors=cfg.shape_anchors)
    ip_out = amplitude_scale * ip_base * envelope
    if cfg.start_value is not None:
        ip_out = ip_out - float(ip_out[0]) + float(cfg.start_value)
    t_out = np.linspace(0.0, target_duration, rows, dtype=float)
    return IpReferenceTrajectory(
        t_out,
        ip_out,
        template_shot_id=template.shot_id,
        amplitude_scale=amplitude_scale,
        duration_scale=duration_scale,
    )


def generate_ip_reference_from_templates(
    templates: list[IpTemplate],
    *,
    seed: int,
    config: SyntheticIpConfig | None = None,
) -> IpReferenceTrajectory:
    """Выбрать шаблон по seed и построить синтетическую Ip-траекторию."""
    if not templates:
        raise ValueError("At least one Ip template is required")
    rng = np.random.default_rng(int(seed))
    template = templates[int(rng.integers(0, len(templates)))]
    return generate_ip_reference_from_template(template, rng=rng, config=config)


def generate_segmented_ip_reference(*, seed: int, config: SegmentedIpConfig) -> IpReferenceTrajectory:
    """Generate a continuous segment-based Ip trajectory."""
    rng = np.random.default_rng(int(seed))
    lo, hi = config.value_bounds
    current = float(rng.uniform(lo, hi)) if config.start_value is None else float(config.start_value)
    values: list[float] = [current]
    segments: list[IpSegment] = []
    count_lo, count_hi = config.segment_count_bounds
    segment_count = int(rng.integers(count_lo, count_hi + 1))
    for _ in range(segment_count):
        if len(values) >= int(config.max_steps):
            break
        duration = int(rng.integers(config.segment_step_bounds[0], config.segment_step_bounds[1] + 1))
        duration = min(duration, int(config.max_steps) - len(values))
        if duration <= 0:
            break
        start_step = len(values) - 1
        if float(rng.random()) < float(config.hold_probability):
            target = current
            kind = "hold"
        else:
            raw_target = float(rng.uniform(lo, hi))
            target = _rate_limited_ip_target(current, raw_target, duration_steps=duration, config=config)
            kind = "ramp"
        if kind == "hold":
            values.extend([current] * duration)
        else:
            for i in range(duration):
                alpha = float(i + 1) / float(duration)
                values.append(float((1.0 - alpha) * current + alpha * target))
        segments.append(
            IpSegment(
                kind=kind,
                start_step=start_step,
                end_step=len(values) - 1,
                start_value=current,
                end_value=target,
            )
        )
        current = float(target)
    if len(values) < 2:
        values.append(current)
    time_s = np.arange(len(values), dtype=float) * float(config.t_step)
    return IpReferenceTrajectory(time_s, np.asarray(values, dtype=float), template_shot_id="segmented", segments=tuple(segments))


def write_semicolon_ip_table(path: str | Path, trajectory: IpReferenceTrajectory) -> None:
    """Записать Ip reference в headerless CSV с разделителем `;`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, trajectory.table(), delimiter=";", fmt="%.16g")


def _sample_scale(rng: np.random.Generator, *, jitter: float) -> float:
    """Сэмплировать положительный масштаб около 1.0."""
    if float(jitter) <= 0.0:
        return 1.0
    scale = 1.0 + float(rng.normal(0.0, float(jitter)))
    return max(scale, 0.05)


def _shape_envelope(
    n_rows: int,
    rng: np.random.Generator,
    *,
    shape_jitter: float,
    n_anchors: int = 5,
) -> np.ndarray:
    """Собрать гладкую мультипликативную огибающую формы сигнала."""
    if int(n_rows) < 2:
        raise ValueError(f"n_rows must be >= 2, got {n_rows}")
    anchors = int(n_anchors)
    if anchors < 2:
        raise ValueError("n_anchors must be >= 2")
    if float(shape_jitter) <= 0.0:
        return np.ones((int(n_rows),), dtype=float)

    anchors_x = np.linspace(0.0, 1.0, anchors, dtype=float)
    anchors_y = rng.normal(0.0, float(shape_jitter), size=anchors)
    anchors_y[0] = 0.0
    anchors_y[-1] = 0.0
    query_x = np.linspace(0.0, 1.0, int(n_rows), dtype=float)
    envelope = 1.0 + np.interp(query_x, anchors_x, anchors_y)
    return np.clip(envelope, 0.1, None)


def _rate_limited_ip_target(current: float, raw_target: float, *, duration_steps: int, config: SegmentedIpConfig) -> float:
    target = float(np.clip(raw_target, config.value_bounds[0], config.value_bounds[1]))
    if config.rate_limit is None:
        return target
    max_delta = float(config.rate_limit) * float(config.t_step) * float(duration_steps)
    delta = float(np.clip(target - float(current), -max_delta, max_delta))
    return float(float(current) + delta)
