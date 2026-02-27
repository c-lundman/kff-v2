from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BANKED_ROOT = REPO_ROOT / "data" / "synthetic"


def test_perfect_banked_has_event_level_file() -> None:
    path = BANKED_ROOT / "perfect_banked" / "multi_2026-01-20_2026-01-22" / "events.csv"
    df = pd.read_csv(path, nrows=5)
    assert {"pax_id", "in_ts_utc", "out_ts_utc", "wait_seconds"} <= set(df.columns)


def test_lossy_banked_has_consistent_measured_event_schema() -> None:
    variants = ["banked_mild", "banked_asymmetric_in"]
    required = {"event_id", "direction", "ts_utc", "source_pax_id", "corruption"}
    for variant in variants:
        path = BANKED_ROOT / "lossy_banked" / "multi_2026-01-20_2026-01-22" / variant / "measured_events.csv"
        df = pd.read_csv(path, nrows=5)
        assert required <= set(df.columns)

