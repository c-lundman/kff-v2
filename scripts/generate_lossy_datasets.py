#!/usr/bin/env python3
"""Create lossy measurement variants from a perfect synthetic dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Dict, List, Tuple


@dataclass
class MeasuredEvent:
    direction: str  # "in" or "out"
    ts: datetime
    source_pax_id: str  # empty for spurious events
    corruption: str  # "original" or "spurious"


def _parse_iso_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def load_perfect_events(path: Path) -> Tuple[List[datetime], List[datetime]]:
    in_times: List[datetime] = []
    out_times: List[datetime] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            in_times.append(_parse_iso_z(row["in_ts_utc"]))
            out_times.append(_parse_iso_z(row["out_ts_utc"]))
    return in_times, out_times


def build_measured_stream(
    in_times: List[datetime],
    out_times: List[datetime],
    *,
    seed: int,
    miss_in_rate: float,
    miss_out_rate: float,
    spurious_in_rate: float,
    spurious_out_rate: float,
    jitter_std_seconds: float,
    day_start: datetime,
    day_end: datetime,
) -> List[MeasuredEvent]:
    rng = random.Random(seed)

    def jitter(ts: datetime) -> datetime:
        if jitter_std_seconds <= 0.0:
            return ts
        t = ts + timedelta(seconds=rng.gauss(0.0, jitter_std_seconds))
        if t < day_start:
            return day_start
        if t > day_end:
            return day_end
        return t

    events: List[MeasuredEvent] = []

    for idx, ts in enumerate(in_times, start=1):
        if rng.random() >= miss_in_rate:
            events.append(
                MeasuredEvent(
                    direction="in",
                    ts=jitter(ts),
                    source_pax_id=str(idx),
                    corruption="original",
                )
            )

    for idx, ts in enumerate(out_times, start=1):
        if rng.random() >= miss_out_rate:
            events.append(
                MeasuredEvent(
                    direction="out",
                    ts=jitter(ts),
                    source_pax_id=str(idx),
                    corruption="original",
                )
            )

    duration_seconds = max(1.0, (day_end - day_start).total_seconds())
    n_spur_in = int(round(len(in_times) * spurious_in_rate))
    n_spur_out = int(round(len(out_times) * spurious_out_rate))

    for _ in range(n_spur_in):
        ts = day_start + timedelta(seconds=rng.random() * duration_seconds)
        events.append(MeasuredEvent(direction="in", ts=ts, source_pax_id="", corruption="spurious"))
    for _ in range(n_spur_out):
        ts = day_start + timedelta(seconds=rng.random() * duration_seconds)
        events.append(MeasuredEvent(direction="out", ts=ts, source_pax_id="", corruption="spurious"))

    events.sort(key=lambda e: e.ts)
    return events


def aggregate_minute_flows(
    events: List[MeasuredEvent], day_start: datetime, day_end: datetime
) -> List[Tuple[datetime, int, int, int]]:
    start_minute = day_start.replace(second=0, microsecond=0)
    end_minute = day_end.replace(second=0, microsecond=0) + timedelta(minutes=1)
    n_minutes = int((end_minute - start_minute).total_seconds() // 60)
    in_counts = [0] * n_minutes
    out_counts = [0] * n_minutes

    for ev in events:
        k = int((ev.ts - start_minute).total_seconds() // 60)
        if 0 <= k < n_minutes:
            if ev.direction == "in":
                in_counts[k] += 1
            else:
                out_counts[k] += 1

    rows: List[Tuple[datetime, int, int, int]] = []
    occ = 0
    for k in range(n_minutes):
        occ += in_counts[k] - out_counts[k]
        rows.append((start_minute + timedelta(minutes=k), in_counts[k], out_counts[k], occ))
    return rows


def write_variant(
    variant_dir: Path,
    events: List[MeasuredEvent],
    minute_rows: List[Tuple[datetime, int, int, int]],
    variant_name: str,
    config: Dict[str, float],
    ground_truth_counts: Dict[str, int],
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)

    with (variant_dir / "measured_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "direction", "ts_utc", "source_pax_id", "corruption"])
        for idx, ev in enumerate(events, start=1):
            writer.writerow([idx, ev.direction, _iso_z(ev.ts), ev.source_pax_id, ev.corruption])

    with (variant_dir / "minute_flows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["minute_start_utc", "in_count", "out_count", "naive_occupancy_end"])
        for ts, in_c, out_c, occ in minute_rows:
            writer.writerow([_iso_z(ts), in_c, out_c, occ])

    occ_series = [r[3] for r in minute_rows]
    neg_minutes = sum(1 for x in occ_series if x < 0)
    summary = {
        "variant": variant_name,
        "config": config,
        "counts": {
            "ground_truth_in": ground_truth_counts["in"],
            "ground_truth_out": ground_truth_counts["out"],
            "measured_in": sum(1 for e in events if e.direction == "in"),
            "measured_out": sum(1 for e in events if e.direction == "out"),
            "spurious_in": sum(1 for e in events if e.direction == "in" and e.corruption == "spurious"),
            "spurious_out": sum(1 for e in events if e.direction == "out" and e.corruption == "spurious"),
        },
        "naive_occupancy": {
            "min": min(occ_series) if occ_series else 0,
            "max": max(occ_series) if occ_series else 0,
            "negative_minutes": neg_minutes,
        },
    }

    with (variant_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main() -> None:
    perfect_dir = Path("data/synthetic/perfect/day_2026-01-15")
    perfect_events_path = perfect_dir / "events.csv"
    if not perfect_events_path.exists():
        raise FileNotFoundError(
            f"Missing perfect dataset at {perfect_events_path}. "
            "Run scripts/generate_perfect_dataset.py first."
        )

    in_times, out_times = load_perfect_events(perfect_events_path)
    day_start = min(in_times).replace(second=0, microsecond=0)
    day_end = max(out_times)

    variants: Dict[str, Dict[str, float]] = {
        "mild_noise": {
            "seed": 101,
            "miss_in_rate": 0.05,
            "miss_out_rate": 0.01,
            "spurious_in_rate": 0.01,
            "spurious_out_rate": 0.01,
            "jitter_std_seconds": 2.0,
        },
        "asymmetric_inflow_loss": {
            "seed": 102,
            "miss_in_rate": 0.22,
            "miss_out_rate": 0.01,
            "spurious_in_rate": 0.00,
            "spurious_out_rate": 0.01,
            "jitter_std_seconds": 3.0,
        },
        "spurious_outflow": {
            "seed": 103,
            "miss_in_rate": 0.03,
            "miss_out_rate": 0.00,
            "spurious_in_rate": 0.00,
            "spurious_out_rate": 0.05,
            "jitter_std_seconds": 1.5,
        },
        "mixed_heavy_noise": {
            "seed": 104,
            "miss_in_rate": 0.18,
            "miss_out_rate": 0.05,
            "spurious_in_rate": 0.02,
            "spurious_out_rate": 0.03,
            "jitter_std_seconds": 6.0,
        },
    }

    root_out = Path("data/synthetic/lossy/day_2026-01-15")
    for name, cfg in variants.items():
        events = build_measured_stream(
            in_times,
            out_times,
            seed=int(cfg["seed"]),
            miss_in_rate=cfg["miss_in_rate"],
            miss_out_rate=cfg["miss_out_rate"],
            spurious_in_rate=cfg["spurious_in_rate"],
            spurious_out_rate=cfg["spurious_out_rate"],
            jitter_std_seconds=cfg["jitter_std_seconds"],
            day_start=day_start,
            day_end=day_end,
        )
        minute_rows = aggregate_minute_flows(events, day_start=day_start, day_end=day_end)
        write_variant(
            root_out / name,
            events,
            minute_rows,
            variant_name=name,
            config=cfg,
            ground_truth_counts={"in": len(in_times), "out": len(out_times)},
        )

    print(f"Wrote lossy variants to: {root_out}")
    print(f"Variants: {', '.join(sorted(variants.keys()))}")


if __name__ == "__main__":
    main()
