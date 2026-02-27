import pandas as pd

from kff_v2 import estimate_queue_from_timestamps


def test_estimate_queue_output_schema() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:05Z",
                "2026-01-20T06:00:35Z",
                "2026-01-20T06:01:12Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:45Z",
                "2026-01-20T06:01:40Z",
            ]
        }
    )

    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert queue.index.name == "Tid"
    assert list(queue.columns) == ["Pax i kö", "Pax in i kö", "Pax ur kö"]


def test_estimate_queue_debug_mode_returns_tuple() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})

    queue, debug = estimate_queue_from_timestamps(in_df, out_df, return_debug=True)
    assert queue.index.name == "Tid"
    assert "in_count_measured" in debug.columns
    assert "out_count_measured" in debug.columns
    assert "in_episode" in debug.columns

