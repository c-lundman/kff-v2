#!/usr/bin/env python3
"""Generate timestamp datasets for a single-departure-flight scenario."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import pandas as pd


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def generate_single_departure(
    *,
    seed: int = 456,
    n_pax: int = 200,
    fwhm_minutes: float = 30.0,
    service_pax_per_min: float = 5.0,
    arrival_center_min: float = 60.0,
    horizon_min: float = 180.0,
) -> tuple[list[datetime], list[datetime], dict]:
    rng = random.Random(seed)
    t0 = datetime(2026, 2, 2, 8, 0, 0, tzinfo=timezone.utc)

    # Gaussian arrivals with desired FWHM.
    sigma_min = fwhm_minutes / 2.354820045
    in_times: list[datetime] = []
    for _ in range(n_pax):
        x = rng.gauss(arrival_center_min, sigma_min)
        x = min(max(x, 0.0), horizon_min)
        in_times.append(t0 + timedelta(minutes=x))
    in_times.sort()

    # FIFO service with approximately constant capacity.
    service_sec = 60.0 / service_pax_per_min
    prev_departure = t0
    out_times: list[datetime] = []
    for in_ts in in_times:
        start_service = max(in_ts, prev_departure)
        out_ts = start_service + timedelta(seconds=service_sec)
        out_times.append(out_ts)
        prev_departure = out_ts

    meta = {
        "scenario": "single_departure_flight",
        "seed": seed,
        "n_pax": n_pax,
        "arrival_fwhm_min": fwhm_minutes,
        "arrival_center_min": arrival_center_min,
        "service_pax_per_min": service_pax_per_min,
        "start_ts_utc": _iso_z(t0),
        "end_ts_utc": _iso_z(out_times[-1]),
    }
    return in_times, out_times, meta


def write_events(path: Path, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [_iso_z(ts) for ts in timestamps]}).to_csv(path, index=False)


def make_rpc50l(in_times: list[datetime], seed: int = 999) -> list[datetime]:
    rng = random.Random(seed)
    return [ts for ts in in_times if rng.random() >= 0.5]


def main() -> None:
    in_times, out_times, meta = generate_single_departure()

    root = Path("data/scenarios/single_departure_flight")
    write_events(root / "PPC_in" / "events.csv", in_times)
    write_events(root / "PPC_out" / "events.csv", out_times)
    write_events(root / "RPC50L_in" / "events.csv", make_rpc50l(in_times))

    with (root / "scenario.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote scenario to: {root}")
    print(f"PPC in events: {len(in_times)}")
    print(f"PPC out events: {len(out_times)}")
    print(f"RPC50L in events: {len(make_rpc50l(in_times))}")


if __name__ == "__main__":
    main()

