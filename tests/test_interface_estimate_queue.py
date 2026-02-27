import pandas as pd

from kff_v2 import EstimateQueueOptions, estimate_queue_from_timestamps


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
    assert list(queue.columns) == ["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"]
    assert queue.index.tz is None


def test_estimate_queue_debug_mode_returns_tuple() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})

    queue, debug = estimate_queue_from_timestamps(in_df, out_df, return_debug=True)
    assert queue.index.name == "Tid"
    assert "in_count_measured" in debug.columns
    assert "out_count_measured" in debug.columns
    assert "in_episode" in debug.columns
    assert "Väntetid" in debug.columns


def test_estimate_queue_can_include_fifo_wait_column() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    opts = EstimateQueueOptions(include_fifo_wait=True)
    queue = estimate_queue_from_timestamps(in_df, out_df, options=opts)
    assert "Väntetid" in queue.columns


def test_estimate_queue_can_disable_fifo_wait_column() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    opts = EstimateQueueOptions(include_fifo_wait=False)
    queue = estimate_queue_from_timestamps(in_df, out_df, options=opts)
    assert "Väntetid" not in queue.columns


def test_estimate_queue_small_case_has_consistent_occupancy() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:01Z",
                "2026-01-20T06:00:10Z",
                "2026-01-20T06:00:40Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:03:02Z",
                "2026-01-20T06:03:11Z",
                "2026-01-20T06:03:59Z",
            ]
        }
    )
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert queue["Pax in i kö"].tolist() == [3.0, 0.0, 0.0, 0.0]
    assert queue["Pax ur kö"].tolist() == [0.0, 0.0, 0.0, 3.0]
    assert queue["Pax i kö"].tolist() == [3.0, 3.0, 3.0, 0.0]
    waits = queue["Väntetid"].tolist()
    assert pd.isna(waits[0]) and pd.isna(waits[1]) and pd.isna(waits[2])
    assert waits[3] == 3.0
