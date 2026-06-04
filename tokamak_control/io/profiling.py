# tokamak_control/io/profiling.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
import logging
import time
from pathlib import Path


class Profiler:
    """Small opt-in timing helper kept separate from normal logging."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        summary_every: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.summary_every = int(summary_every)
        self.logger = logger
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.path_counts: dict[str, int] = defaultdict(int)
        self.step_counter: int = 0

    def configure(
        self,
        *,
        enabled: bool | None = None,
        summary_every: int | None = None,
        logger: logging.Logger | None = None,
        reset: bool = False,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if summary_every is not None:
            self.summary_every = int(summary_every)
        if logger is not None:
            self.logger = logger
        if reset:
            self.reset()

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.path_counts.clear()
        self.step_counter = 0

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] += dt
            self.counts[name] += 1

    def record_path(self, name: str) -> None:
        if self.enabled:
            self.path_counts[name] += 1

    def step(self) -> None:
        if self.enabled:
            self.step_counter += 1

    def should_report(self, total_key: str) -> bool:
        if not self.enabled or self.summary_every <= 0:
            return False
        total_calls = int(self.counts.get(total_key, 0))
        return total_calls > 0 and total_calls % self.summary_every == 0

    def summary_dict(
        self,
        *,
        total_key: str | None = None,
        keys: Sequence[str] = (),
        path_keys: Sequence[str] = (),
        title: str | None = None,
    ) -> dict[str, object]:
        block_names = list(keys) if keys else sorted(self.totals.keys())
        path_names = list(path_keys) if path_keys else sorted(self.path_counts.keys())

        total_time = None
        total_calls = None
        total_mean_ms = None
        if total_key is not None:
            total_time = float(self.totals.get(total_key, 0.0))
            total_calls = int(self.counts.get(total_key, 0))
            total_mean_ms = 1e3 * total_time / max(total_calls, 1)

        blocks: dict[str, dict[str, float | int]] = {}
        denom = float(total_time) if total_time is not None else 0.0
        for key in block_names:
            t = float(self.totals.get(key, 0.0))
            calls = int(self.counts.get(key, 0))
            mean_ms = 1e3 * t / max(calls, 1)
            block = {
                "total_s": t,
                "calls": calls,
                "mean_ms": mean_ms,
            }
            if total_key is not None:
                block["pct_of_total"] = 100.0 * t / denom if denom > 0.0 else 0.0
            blocks[key] = block

        return {
            "title": title,
            "enabled": bool(self.enabled),
            "summary_every": int(self.summary_every),
            "step_counter": int(self.step_counter),
            "total_key": total_key,
            "total": (
                None
                if total_key is None
                else {
                    "name": total_key,
                    "total_s": float(total_time),
                    "calls": int(total_calls),
                    "mean_ms": float(total_mean_ms),
                }
            ),
            "blocks": blocks,
            "paths": {key: int(self.path_counts.get(key, 0)) for key in path_names},
        }

    def log_summary(
        self,
        *,
        total_key: str,
        keys: Sequence[str],
        title: str,
        path_keys: Sequence[str] = (),
    ) -> None:
        if self.logger is None or not self.should_report(total_key):
            return
        summary = self.summary_dict(
            total_key=total_key,
            keys=keys,
            path_keys=path_keys,
            title=title,
        )
        total = summary["total"]
        if not isinstance(total, Mapping):
            return
        self.logger.info("[%s] %s: total=%.3fs calls=%d mean=%.3fms", title, total["name"], total["total_s"], total["calls"], total["mean_ms"])
        for key in keys:
            block = summary["blocks"].get(key, {"total_s": 0.0, "pct_of_total": 0.0, "calls": 0, "mean_ms": 0.0})
            self.logger.info("[%s] %s: total=%.3fs pct=%.1f%% calls=%d mean=%.3fms", title, key, block["total_s"], block.get("pct_of_total", 0.0), block["calls"], block["mean_ms"])
        if path_keys:
            parts = [f"{k}={int(summary['paths'].get(k, 0))}" for k in path_keys]
            self.logger.info("[%s] paths: %s", title, " ".join(parts))

    def write_json(
        self,
        path: str | Path,
        *,
        total_key: str | None = None,
        keys: Sequence[str] = (),
        path_keys: Sequence[str] = (),
        title: str | None = None,
    ) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary_dict(
            total_key=total_key,
            keys=keys,
            path_keys=path_keys,
            title=title,
        )
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
