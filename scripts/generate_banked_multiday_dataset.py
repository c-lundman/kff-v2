#!/usr/bin/env python3
"""Generate multi-day synthetic data with spiky flight-bank traffic."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import numpy as np
import pandas as pd


def _minute_range(start: datetime, end: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="1min", tz="UTC")


def _daily_capacity(minute_ts: pd.Timestamp, rng: random.Random) -> float:
    h = minute_ts.hour + minute_ts.minute / 60.0
    if 5.5 <= h <= 10.0:
        base = 9.5
    elif 14.0 <= h <= 19.0:
        base = 9.0
    else:
        base = 4.0
    return max(0.0, base + rng.uniform(-0.8, 0.8))


def _add_flight_burst(inflow: np.ndarray, idx0: int, pax: int, width_min: int = 40) -> None:
    # Bell-like unloading profile across width_min minutes.
    x = np.linspace(-2.5, 2.5, width_min)
    weights = np.exp(-0.5 * x * x)
    weights = weights / weights.sum()
    alloc = np.floor(weights * pax).astype(int)
    residual = pax - int(alloc.sum())
    if residual > 0:
        alloc[np.argsort(-weights)[:residual]] += 1
    end = min(len(inflow), idx0 + width_min)
    inflow[idx0:end] += alloc[: end - idx0]


def generate_dataset(seed: int = 2026) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)

    start = datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 22, 23, 59, tzinfo=timezone.utc)
    ts = _minute_range(start, end)
    n = len(ts)

    inflow = np.zeros(n, dtype=int)
    flights = []

    current_day = start.date()
    while current_day <= end.date():
        day_start = datetime.combine(current_day, datetime.min.time(), tzinfo=timezone.utc)
        day_idx0 = int((day_start - start).total_seconds() // 60)

        # Morning and afternoon banks + sparse isolated flights.
        n_morning = rng.randint(5, 10)
        n_afternoon = rng.randint(5, 10)
        n_sparse = rng.randint(1, 4)

        def schedule_window(start_hour: float, end_hour: float, n_flights: int, tag: str) -> None:
            for _ in range(n_flights):
                dep_min = int(rng.uniform(start_hour * 60, end_hour * 60))
                pax = rng.randint(120, 260) if tag != "sparse" else rng.randint(80, 180)
                idx = day_idx0 + dep_min
                if 0 <= idx < n:
                    _add_flight_burst(inflow, idx, pax, width_min=rng.randint(30, 50))
                    flights.append(
                        {
                            "day": str(current_day),
                            "bank": tag,
                            "scheduled_minute_utc": str(ts[idx]),
                            "pax": pax,
                        }
                    )

        schedule_window(6.0, 8.0, n_morning, "morning")
        schedule_window(15.0, 17.0, n_afternoon, "afternoon")
        schedule_window(10.0, 21.0, n_sparse, "sparse")

        current_day = current_day + timedelta(days=1)

    outflow = np.zeros(n, dtype=int)
    occupancy = np.zeros(n, dtype=int)
    q = 0.0
    for k, t in enumerate(ts):
        q += float(inflow[k])
        cap = _daily_capacity(t, rng)
        served = min(q, max(0.0, cap + rng.gauss(0.0, 0.6)))
        served_i = int(round(served))
        served_i = max(0, min(int(round(q)), served_i))
        q -= served_i
        outflow[k] = served_i
        occupancy[k] = int(round(q))

    perfect = pd.DataFrame(
        {
            "minute_start_utc": ts,
            "in_count": inflow,
            "out_count": outflow,
            "occupancy_end": occupancy,
        }
    )

    summary = {
        "scenario": "banked_multiday_v1",
        "seed": seed,
        "start_ts_utc": str(ts.min()),
        "end_ts_utc": str(ts.max()),
        "num_minutes": int(n),
        "num_flights": len(flights),
        "flights": flights,
    }
    return perfect, summary


def make_lossy_variants(perfect: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = random.Random(77)
    variants: dict[str, tuple[float, float, float, float]] = {
        "banked_mild": (0.04, 0.01, 0.01, 0.01),
        "banked_asymmetric_in": (0.18, 0.01, 0.00, 0.01),
    }
    out: dict[str, pd.DataFrame] = {}
    for name, (miss_in, miss_out, spur_in, spur_out) in variants.items():
        df = perfect.copy()
        in_meas = []
        out_meas = []
        for i, o in zip(df["in_count"].to_numpy(), df["out_count"].to_numpy()):
            i_m = sum(1 for _ in range(int(i)) if rng.random() >= miss_in)
            o_m = sum(1 for _ in range(int(o)) if rng.random() >= miss_out)
            i_m += sum(1 for _ in range(max(1, int(i))) if rng.random() < spur_in)
            o_m += sum(1 for _ in range(max(1, int(o))) if rng.random() < spur_out)
            in_meas.append(i_m)
            out_meas.append(o_m)
        df = df[["minute_start_utc"]].copy()
        df["in_count"] = in_meas
        df["out_count"] = out_meas
        df["naive_occupancy_end"] = (df["in_count"] - df["out_count"]).cumsum()
        out[name] = df
    return out


def main() -> None:
    perfect, summary = generate_dataset()

    root = Path("data/synthetic/perfect_banked/multi_2026-01-20_2026-01-22")
    root.mkdir(parents=True, exist_ok=True)
    perfect.to_csv(root / "minute_flows.csv", index=False)
    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    lossy = make_lossy_variants(perfect)
    lossy_root = Path("data/synthetic/lossy_banked/multi_2026-01-20_2026-01-22")
    for name, df in lossy.items():
        out_dir = lossy_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "minute_flows.csv", index=False)
        s = {
            "variant": name,
            "naive_occupancy_min": float(df["naive_occupancy_end"].min()),
            "naive_negative_minutes": int((df["naive_occupancy_end"] < 0).sum()),
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
            f.write("\n")

    print(f"Wrote perfect banked dataset to: {root}")
    print(f"Wrote lossy banked variants to: {lossy_root}")


if __name__ == "__main__":
    main()

