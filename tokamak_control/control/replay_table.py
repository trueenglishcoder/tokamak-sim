"""Утилиты чтения и интерполяции таблиц воспроизведения токов."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_numeric_table(path: Path) -> np.ndarray:
    """Загрузить числовую CSV/TXT-таблицу с автоопределением разделителя."""
    lines: list[str] = []
    delimiter: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            if delimiter is None:
                delimiter = _detect_delimiter(line)

            parts = line.split(delimiter) if delimiter is not None else line.split()
            parts = [part.strip() for part in parts if part.strip() != ""]
            if _is_numeric_row(parts):
                lines.append(" ".join(parts))

    if not lines:
        raise ValueError(f"No numeric rows found in replay table: {path}")

    rows = [np.fromstring(line, sep=" ", dtype=float) for line in lines]
    return _stack_rows(rows, path)


def coalesce_near_duplicate_times(table: np.ndarray, *, time_eps: float) -> np.ndarray:
    """Объединить соседние строки с практически одинаковым временем."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0:
        return table

    rows: list[np.ndarray] = []
    group: list[np.ndarray] = [table[0]]
    last_t = float(table[0, 0])

    for row in table[1:]:
        t = float(row[0])
        dt = t - last_t
        if dt < -time_eps:
            raise ValueError(
                f"Replay table time column must be nondecreasing. Found {last_t} followed by {t}"
            )
        if dt <= time_eps:
            group.append(row)
        else:
            rows.append(np.mean(np.vstack(group), axis=0))
            group = [row]
        last_t = t

    rows.append(np.mean(np.vstack(group), axis=0))
    return np.vstack(rows)


def interp_columns_clamped(query_t: float, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Интерполировать столбцы таблицы по времени с зажимом на границах."""
    tq = float(np.clip(query_t, float(t[0]), float(t[-1])))
    out = np.empty((y.shape[1],), dtype=float)
    for j in range(y.shape[1]):
        out[j] = float(np.interp(tq, t, y[:, j]))
    return out


def _detect_delimiter(line: str) -> str | None:
    """Определить разделитель первой числовой строки."""
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    return None


def _is_numeric_row(parts: list[str]) -> bool:
    """Вернуть True, если все ячейки строки можно прочитать как float."""
    if not parts:
        return False
    try:
        [float(part) for part in parts]
    except ValueError:
        return False
    return True


def _stack_rows(rows: list[np.ndarray], path: Path) -> np.ndarray:
    """Собрать строки одинаковой ширины в двумерный массив."""
    width = int(rows[0].size)
    if width == 0:
        raise ValueError(f"Could not parse replay table: {path}")
    for i, row in enumerate(rows, start=1):
        if row.size != width:
            raise ValueError(
                f"Replay table has inconsistent row width at numeric row {i}: "
                f"expected {width}, got {row.size}"
            )
    return np.vstack(rows)

