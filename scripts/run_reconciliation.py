#!/usr/bin/env python3
"""Run QP reconciliation on all lossy variants for a synthetic day."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kff_v2 import ReconcileConfig, reconcile_minute_flows


def main() -> None:
    day = "day_2026-01-15"
    lossy_root = Path("data/synthetic/lossy") / day
    perfect_path = Path("data/synthetic/perfect") / day / "minute_flows.csv"
    out_root = Path("data/synthetic/reconciled") / day
    out_root.mkdir(parents=True, exist_ok=True)

    perfect = pd.read_csv(perfect_path)[["minute_start_utc", "occupancy_end"]].rename(
        columns={"occupancy_end": "occupancy_truth_end"}
    )

    config = ReconcileConfig(
        q0=0.0,
        w_in=1.0,
        w_out=10.0,
        smooth_in=0.05,
        smooth_out=0.05,
    )

    for variant_dir in sorted([p for p in lossy_root.iterdir() if p.is_dir()]):
        input_path = variant_dir / "minute_flows.csv"
        measured = pd.read_csv(input_path)

        reconciled = reconcile_minute_flows(measured, config=config)
        merged = reconciled.merge(perfect, on="minute_start_utc", how="left")
        merged["occupancy_abs_err"] = (
            merged["occupancy_corrected_end"] - merged["occupancy_truth_end"]
        ).abs()

        out_dir = out_root / variant_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir / "reconciled_minute_flows.csv", index=False)

        summary = {
            "variant": variant_dir.name,
            "config": {
                "q0": config.q0,
                "w_in": config.w_in,
                "w_out": config.w_out,
                "smooth_in": config.smooth_in,
                "smooth_out": config.smooth_out,
            },
            "naive_occupancy": {
                "negative_minutes": int((measured["naive_occupancy_end"] < 0).sum()),
                "min": float(measured["naive_occupancy_end"].min()),
            },
            "reconciled_occupancy": {
                "negative_minutes": int((merged["occupancy_corrected_end"] < -1e-6).sum()),
                "min": float(merged["occupancy_corrected_end"].min()),
                "mae_vs_truth": float(merged["occupancy_abs_err"].mean()),
                "p95_abs_err_vs_truth": float(merged["occupancy_abs_err"].quantile(0.95)),
            },
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    print(f"Wrote reconciled outputs to: {out_root}")


if __name__ == "__main__":
    main()

