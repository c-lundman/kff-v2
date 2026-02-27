#!/usr/bin/env python3
"""Generate a deterministic synthetic ground-truth queue dataset.

Outputs:
- events.csv: one row per pax with in/out timestamps and wait time.
- minute_flows.csv: 1-minute in/out counts and end-of-minute occupancy.
- summary.json: basic scenario metadata and KPIs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import List


@dataclass
class PaxEvent:
    pax_id: int
    in_ts: datetime
    out_ts: datetime

    @property
    def wait_seconds(self) -> float:
        return (self.out_ts - self.in_ts).total_seconds()


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def generate_perfect_fifo_day(
    day_start_utc: datetime,
    duration_hours: int = 16,
    seed: int = 7,
) -> List[PaxEvent]:
    rng = random.Random(seed)

    duration_seconds = duration_hours * 3600

    # Time-varying arrivals (piecewise rates in pax/min).
    def arrival_rate_per_minute(t_sec: float) -> float:
        h = t_sec / 3600.0
        if h < 3:
            return 6.0
        if h < 7:
            return 11.0
        if h < 11:
            return 8.0
        if h < 14:
            return 13.0
        return 7.0

    # Time-varying service capacity (pax/min), lower than peak arrivals at times.
    def service_rate_per_minute(t_sec: float) -> float:
        h = t_sec / 3600.0
        if h < 4:
            return 7.5
        if h < 8:
            return 8.5
        if h < 12:
            return 9.5
        return 10.5

    # Sample arrivals as a non-homogeneous Poisson process using thinning.
    max_arrival_rate_per_sec = 13.0 / 60.0
    t = 0.0
    arrivals: List[datetime] = []
    while t < duration_seconds:
        t += rng.expovariate(max_arrival_rate_per_sec)
        if t >= duration_seconds:
            break
        lam_t = arrival_rate_per_minute(t) / 60.0
        if rng.random() < (lam_t / max_arrival_rate_per_sec):
            arrivals.append(day_start_utc + timedelta(seconds=t))

    # FIFO departures with queue and time-varying service rate.
    events: List[PaxEvent] = []
    if not arrivals:
        return events

    prev_departure = arrivals[0]
    for idx, in_ts in enumerate(arrivals, start=1):
        service_mean_sec = 60.0 / service_rate_per_minute((in_ts - day_start_utc).total_seconds())
        service_sec = max(1.0, rng.expovariate(1.0 / service_mean_sec))
        service_dur = timedelta(seconds=service_sec)
        start_service = max(in_ts, prev_departure)
        out_ts = start_service + service_dur
        prev_departure = out_ts
        events.append(PaxEvent(pax_id=idx, in_ts=in_ts, out_ts=out_ts))

    return events


def write_outputs(events: List[PaxEvent], out_dir: Path, day_start: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pax_id", "in_ts_utc", "out_ts_utc", "wait_seconds"])
        for ev in events:
            writer.writerow([ev.pax_id, _iso_z(ev.in_ts), _iso_z(ev.out_ts), round(ev.wait_seconds, 3)])

    # 1-minute binned in/out flows and occupancy.
    if not events:
        return

    start_minute = day_start.replace(second=0, microsecond=0)
    end_ts = max(events[-1].out_ts, events[-1].in_ts)
    end_minute = end_ts.replace(second=0, microsecond=0) + timedelta(minutes=1)
    n_minutes = int((end_minute - start_minute).total_seconds() // 60)

    in_counts = [0] * n_minutes
    out_counts = [0] * n_minutes

    for ev in events:
        i = int((ev.in_ts - start_minute).total_seconds() // 60)
        o = int((ev.out_ts - start_minute).total_seconds() // 60)
        if 0 <= i < n_minutes:
            in_counts[i] += 1
        if 0 <= o < n_minutes:
            out_counts[o] += 1

    minute_path = out_dir / "minute_flows.csv"
    occ = 0
    with minute_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["minute_start_utc", "in_count", "out_count", "occupancy_end"])
        for k in range(n_minutes):
            occ += in_counts[k] - out_counts[k]
            ts = start_minute + timedelta(minutes=k)
            writer.writerow([_iso_z(ts), in_counts[k], out_counts[k], occ])

    waits = sorted(ev.wait_seconds for ev in events)
    def percentile(p: float) -> float:
        if not waits:
            return 0.0
        idx = min(len(waits) - 1, max(0, int(round((p / 100.0) * (len(waits) - 1)))))
        return waits[idx]

    summary = {
        "scenario": "perfect_day_v1",
        "timezone": "UTC",
        "seed": 7,
        "num_pax": len(events),
        "start_ts_utc": _iso_z(start_minute),
        "end_ts_utc": _iso_z(end_minute),
        "wait_seconds": {
            "mean": round(sum(waits) / len(waits), 3),
            "p50": round(percentile(50), 3),
            "p90": round(percentile(90), 3),
            "p95": round(percentile(95), 3),
            "max": round(max(waits), 3),
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main() -> None:
    day_start = datetime(2026, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
    events = generate_perfect_fifo_day(day_start_utc=day_start)
    out_dir = Path("data/synthetic/perfect/day_2026-01-15")
    write_outputs(events, out_dir, day_start)
    print(f"Wrote perfect dataset to: {out_dir}")
    print(f"Generated pax: {len(events)}")


if __name__ == "__main__":
    main()
