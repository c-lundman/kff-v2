#!/usr/bin/env python3
"""Detect queue episodes and run per-episode QP reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kff_v2 import (
    EpisodeDetectConfig,
    ReconcileConfig,
    detect_queue_episodes,
    reconcile_by_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="multi_2026-01-20_2026-01-22")
    parser.add_argument("--variant", default="banked_asymmetric_in")
    parser.add_argument("--all", action="store_true", help="Run for all available lossy variants.")
    return parser.parse_args()


def run_one(day_key: str, variant: str) -> None:
    input_path = Path("data/synthetic/lossy_banked") / day_key / variant / "minute_flows.csv"
    perfect_path = Path("data/synthetic/perfect_banked") / day_key / "minute_flows.csv"
    out_dir = Path("data/synthetic/reconciled_banked") / day_key / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = pd.read_csv(input_path)
    perfect = pd.read_csv(perfect_path)[["minute_start_utc", "occupancy_end"]].rename(
        columns={"occupancy_end": "occupancy_truth_end"}
    )

    ep_cfg = EpisodeDetectConfig(
        active_threshold=1.0,
        min_active_minutes=8,
        max_gap_minutes=12,
        min_episode_minutes=30,
        buffer_minutes=12,
    )
    rec_cfg = ReconcileConfig(q0=0.0, w_in=1.0, w_out=12.0, smooth_in=0.05, smooth_out=0.05)

    episodes = detect_queue_episodes(measured, config=ep_cfg)
    rec = reconcile_by_episodes(measured, reconcile_config=rec_cfg, episode_config=ep_cfg)
    merged = rec.merge(perfect, on="minute_start_utc", how="left")
    merged["occupancy_abs_err"] = (
        merged["occupancy_corrected_end"] - merged["occupancy_truth_end"]
    ).abs()
    merged["naive_occupancy_end"] = (merged["in_count_measured"] - merged["out_count_measured"]).cumsum()

    merged.to_csv(out_dir / "reconciled_minute_flows.csv", index=False)
    episodes.to_csv(out_dir / "episodes.csv", index=False)

    summary = {
        "variant": variant,
        "num_episodes": int(len(episodes)),
        "minutes_in_episodes": int(merged["in_episode"].sum()),
        "naive_negative_minutes": int((merged["naive_occupancy_end"] < 0).sum()),
        "reconciled_negative_minutes": int((merged["occupancy_corrected_end"] < -1e-6).sum()),
        "mae_vs_truth": float(merged["occupancy_abs_err"].mean()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote episode reconciliation outputs to: {out_dir}")
    print(f"Detected episodes: {len(episodes)}")


def main() -> None:
    args = parse_args()
    if args.all:
        lossy_root = Path("data/synthetic/lossy_banked") / args.key
        variants = sorted([p.name for p in lossy_root.iterdir() if p.is_dir()])
        for variant in variants:
            run_one(args.key, variant)
        return
    run_one(args.key, args.variant)


if __name__ == "__main__":
    main()
